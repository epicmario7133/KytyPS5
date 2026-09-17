#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"

#include <functional>
#include <vector>

// Software emulation of image_bvh_intersect_ray / image_bvh64_intersect_ray (RDNA2 RT IP 1.1).
// The hardware tests a ray against a single BVH node held in guest memory:
//   - node types 0..3 are triangle nodes (64 bytes: four fp32 vertices, geometry/primitive
//     indices and a triangle_id word). The result is {t_num, t_denom, i_num, j_num}.
//   - node type 4 is an fp16 box node (64 bytes), type 5 an fp32 box node (128 bytes), each
//     with four children. The result is the four child pointers, INVALID_NODE for misses,
//     sorted closest-first when the descriptor enables box sorting.
//   - other types are user nodes and return all-ones.
// The math follows AMD's own emulation in GPURT (IntersectCommon.hlsl).

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

constexpr uint32_t InvalidNode = 0xffffffffu;

constexpr uint32_t BvhNodeTypeBox16     = 4u;
constexpr uint32_t BvhNodeTypeBox32     = 5u;
constexpr uint32_t TriangleIdOffset     = 60u;
constexpr uint32_t Box32ChildOffset     = 0u;
constexpr uint32_t Box32BoundsOffset    = 16u;
constexpr uint32_t Box32BoundsStride    = 24u;
constexpr uint32_t Box16BoundsOffset    = 16u;
constexpr uint32_t Box16BoundsStride    = 12u;
constexpr uint32_t DescBoxGrowShift     = 23u;
constexpr uint32_t DescBoxSortEnable    = 1u << 31u;
constexpr uint32_t DescTriangleReturnIJ = 1u << 24u;
constexpr float    BoxGrowEpsilon       = 5.960464478e-8f; // 2^-24

struct Vec3 {
	uint32_t x = 0;
	uint32_t y = 0;
	uint32_t z = 0;
};

uint32_t FBin(EmitterState& state, spv::Op op, uint32_t a, uint32_t b) {
	return Binary(state, op, TypeF32(state), a, b);
}

uint32_t FCmp(EmitterState& state, spv::Op op, uint32_t a, uint32_t b) {
	return Binary(state, op, TypeBool(state), a, b);
}

uint32_t FMaxN(EmitterState& state, uint32_t a, uint32_t b) {
	return EmitGlsl<GLSLstd450NMax, IR::Type::F32>(state, a, b);
}

uint32_t FMinN(EmitterState& state, uint32_t a, uint32_t b) {
	return EmitGlsl<GLSLstd450NMin, IR::Type::F32>(state, a, b);
}

uint32_t BoolOr(EmitterState& state, uint32_t a, uint32_t b) {
	return Binary(state, spv::OpLogicalOr, TypeBool(state), a, b);
}

uint32_t BoolAnd(EmitterState& state, uint32_t a, uint32_t b) {
	return Binary(state, spv::OpLogicalAnd, TypeBool(state), a, b);
}

uint32_t SelectF(EmitterState& state, uint32_t cond, uint32_t a, uint32_t b) {
	return Select(state, TypeF32(state), cond, a, b);
}

uint32_t SelectU(EmitterState& state, uint32_t cond, uint32_t a, uint32_t b) {
	return Select(state, TypeU32(state), cond, a, b);
}

uint32_t Extract(EmitterState& state, uint32_t type, uint32_t composite, uint32_t index) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, type, result, composite, index);
	return result;
}

Vec3 VSub(EmitterState& state, Vec3 a, Vec3 b) {
	return {FBin(state, spv::OpFSub, a.x, b.x), FBin(state, spv::OpFSub, a.y, b.y),
	        FBin(state, spv::OpFSub, a.z, b.z)};
}

Vec3 VMul(EmitterState& state, Vec3 a, Vec3 b) {
	return {FBin(state, spv::OpFMul, a.x, b.x), FBin(state, spv::OpFMul, a.y, b.y),
	        FBin(state, spv::OpFMul, a.z, b.z)};
}

Vec3 VCross(EmitterState& state, Vec3 a, Vec3 b) {
	const auto term = [&](uint32_t p, uint32_t q, uint32_t r, uint32_t s) {
		return FBin(state, spv::OpFSub, FBin(state, spv::OpFMul, p, q),
		            FBin(state, spv::OpFMul, r, s));
	};
	return {term(a.y, b.z, a.z, b.y), term(a.z, b.x, a.x, b.z), term(a.x, b.y, a.y, b.x)};
}

uint32_t VDot(EmitterState& state, Vec3 a, Vec3 b) {
	return FBin(state, spv::OpFAdd,
	            FBin(state, spv::OpFAdd, FBin(state, spv::OpFMul, a.x, b.x),
	                 FBin(state, spv::OpFMul, a.y, b.y)),
	            FBin(state, spv::OpFMul, a.z, b.z));
}

uint32_t AddressAt(EmitterState& state, uint32_t base, uint32_t offset) {
	return offset == 0u ? base
	                    : Binary(state, spv::OpIAdd, TypeScalarU64(state), base,
	                             ConstantDeviceAddress(state, offset));
}

// `base` is a host device address (already translated through the BDA page table). Nodes
// are 64-byte aligned and never straddle a caching page, so one translation covers a node.
uint32_t LoadU32At(EmitterState& state, uint32_t base, uint32_t offset) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer,
	                          AddressAt(state, base, offset));
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer,
	                          spv::MemoryAccessAlignedMask, 4u);
	return value;
}

uint32_t LoadF32At(EmitterState& state, uint32_t base, uint32_t offset) {
	return EmitBitcastU32ToF32(state, LoadU32At(state, base, offset));
}

Vec3 LoadVec3At(EmitterState& state, uint32_t base, uint32_t offset) {
	return {LoadF32At(state, base, offset), LoadF32At(state, base, offset + 4u),
	        LoadF32At(state, base, offset + 8u)};
}

// Unpacks a dword of two halves into (low, high) floats.
std::pair<uint32_t, uint32_t> UnpackHalves(EmitterState& state, uint32_t bits) {
	const auto unpacked = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeF32Vector(state, 2), unpacked, GlslStd450(state),
	                          GLSLstd450UnpackHalf2x16, bits);
	return {Extract(state, TypeF32(state), unpacked, 0),
	        Extract(state, TypeF32(state), unpacked, 1)};
}

// Emits `cond ? then() : else()` for a list of values of the given types, returning the phis.
template <typename Then, typename Else>
std::vector<uint32_t> EmitIfElseValues(EmitterState& state, uint32_t condition,
                                       const std::vector<uint32_t>& types, Then&& then_fn,
                                       Else&& else_fn) {
	const auto then_label  = state.builder.AllocateId();
	const auto else_label  = state.builder.AllocateId();
	const auto merge_label = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, condition, then_label, else_label);
	EmitLabel(state, then_label);
	const std::vector<uint32_t> then_values = then_fn();
	const auto                  then_exit   = state.current_label;
	state.builder.AddFunction(spv::OpBranch, merge_label);
	EmitLabel(state, else_label);
	const std::vector<uint32_t> else_values = else_fn();
	const auto                  else_exit   = state.current_label;
	state.builder.AddFunction(spv::OpBranch, merge_label);
	EmitLabel(state, merge_label);
	EXIT_IF(then_values.size() != types.size() || else_values.size() != types.size());
	std::vector<uint32_t> results(types.size());
	for (size_t index = 0; index < types.size(); index++) {
		results[index] = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpPhi, types[index], results[index], then_values[index],
		                          then_exit, else_values[index], else_exit);
	}
	return results;
}

struct BoxNode {
	uint32_t children[4] = {};
	Vec3     min[4];
	Vec3     max[4];
};

BoxNode LoadBox32Node(EmitterState& state, uint32_t node_addr) {
	BoxNode node;
	for (uint32_t i = 0; i < 4u; i++) {
		node.children[i] = LoadU32At(state, node_addr, Box32ChildOffset + i * 4u);
	}
	for (uint32_t i = 0; i < 4u; i++) {
		const auto offset = Box32BoundsOffset + i * Box32BoundsStride;
		node.min[i]       = LoadVec3At(state, node_addr, offset);
		node.max[i]       = LoadVec3At(state, node_addr, offset + 12u);
	}
	return node;
}

BoxNode LoadBox16Node(EmitterState& state, uint32_t node_addr) {
	BoxNode node;
	for (uint32_t i = 0; i < 4u; i++) {
		node.children[i] = LoadU32At(state, node_addr, i * 4u);
	}
	for (uint32_t i = 0; i < 4u; i++) {
		// Each box is three dwords of halves: minx|miny, minz|maxx, maxy|maxz.
		const auto offset     = Box16BoundsOffset + i * Box16BoundsStride;
		const auto [mnx, mny] = UnpackHalves(state, LoadU32At(state, node_addr, offset));
		const auto [mnz, mxx] = UnpackHalves(state, LoadU32At(state, node_addr, offset + 4u));
		const auto [mxy, mxz] = UnpackHalves(state, LoadU32At(state, node_addr, offset + 8u));
		node.min[i]           = {mnx, mny, mnz};
		node.max[i]           = {mxx, mxy, mxz};
	}
	return node;
}

std::vector<uint32_t> FlattenBox(const BoxNode& node) {
	std::vector<uint32_t> values;
	for (uint32_t i = 0; i < 4u; i++) {
		values.push_back(node.children[i]);
	}
	for (uint32_t i = 0; i < 4u; i++) {
		values.insert(values.end(), {node.min[i].x, node.min[i].y, node.min[i].z, node.max[i].x,
		                             node.max[i].y, node.max[i].z});
	}
	return values;
}

BoxNode UnflattenBox(const std::vector<uint32_t>& values) {
	BoxNode node;
	for (uint32_t i = 0; i < 4u; i++) {
		node.children[i] = values[i];
	}
	for (uint32_t i = 0; i < 4u; i++) {
		const auto* v = &values[4u + i * 6u];
		node.min[i]   = {v[0], v[1], v[2]};
		node.max[i]   = {v[3], v[4], v[5]};
	}
	return node;
}

struct BoxSlab {
	uint32_t min_t = 0;
	uint32_t max_t = 0;
};

// GPURT fast_intersect_bbox: slab test with the interval clamped to [0, extent].
BoxSlab IntersectBox(EmitterState& state, Vec3 origin, Vec3 inv_dir, uint32_t extent, Vec3 box_min,
                     Vec3 box_max) {
	const auto zero  = ConstantF32Value(state, 0.0f);
	const auto t_min = VMul(state, VSub(state, box_min, origin), inv_dir);
	const auto t_max = VMul(state, VSub(state, box_max, origin), inv_dir);
	const auto pick  = [&](uint32_t inv, uint32_t lo, uint32_t hi) {
		const auto positive = FCmp(state, spv::OpFOrdGreaterThanEqual, inv, zero);
		return std::pair {SelectF(state, positive, lo, hi), SelectF(state, positive, hi, lo)};
	};
	const auto [min_x, max_x] = pick(inv_dir.x, t_min.x, t_max.x);
	const auto [min_y, max_y] = pick(inv_dir.y, t_min.y, t_max.y);
	const auto [min_z, max_z] = pick(inv_dir.z, t_min.z, t_max.z);
	const auto min_of         = FMaxN(state, FMaxN(state, min_x, min_y), min_z);
	const auto max_of         = FMinN(state, FMinN(state, max_x, max_y), max_z);
	auto       lo             = FMaxN(state, min_of, zero);
	auto       hi             = FMinN(state, max_of, extent);
	const auto nan            = BoolOr(state, Unary(state, spv::OpIsNan, TypeBool(state), min_of),
	                                   Unary(state, spv::OpIsNan, TypeBool(state), max_of));
	lo                        = SelectF(state, nan, ConstantF32(state, 0x7f800000u), lo);
	hi                        = SelectF(state, nan, ConstantF32(state, 0xff800000u), hi);
	return {lo, hi};
}

std::vector<uint32_t> IntersectBoxNode(EmitterState& state, uint32_t node_addr, uint32_t node_type,
                                       uint32_t extent, Vec3 origin, Vec3 inv_dir, uint32_t desc1) {
	std::vector<uint32_t> types(4u, TypeU32(state));
	for (uint32_t i = 0; i < 24u; i++) {
		types.push_back(TypeF32(state));
	}
	const auto is_fp16 = Binary(state, spv::OpIEqual, TypeBool(state), node_type,
	                            ConstantU32(state, BvhNodeTypeBox16));
	const auto node    = UnflattenBox(EmitIfElseValues(
	    state, is_fp16, types, [&] { return FlattenBox(LoadBox16Node(state, node_addr)); },
	    [&] { return FlattenBox(LoadBox32Node(state, node_addr)); }));

	// The ray/box interval grows by box_grow_value ULPs to stay conservative against the
	// ray/triangle test.
	const auto grow = Unary(state, spv::OpConvertUToF, TypeF32(state),
	                        Binary(state, spv::OpBitwiseAnd, TypeU32(state),
	                               Binary(state, spv::OpShiftRightLogical, TypeU32(state), desc1,
	                                      ConstantU32(state, DescBoxGrowShift)),
	                               ConstantU32(state, 0xffu)));
	const auto scale =
	    FBin(state, spv::OpFAdd, ConstantF32Value(state, 1.0f),
	         FBin(state, spv::OpFMul, grow, ConstantF32Value(state, BoxGrowEpsilon)));

	uint32_t traverse[4];
	uint32_t key[4];
	for (uint32_t i = 0; i < 4u; i++) {
		const auto slab = IntersectBox(state, origin, inv_dir, extent, node.min[i], node.max[i]);
		const auto hit  = FCmp(state, spv::OpFOrdLessThanEqual, slab.min_t,
		                       FBin(state, spv::OpFMul, slab.max_t, scale));
		traverse[i]     = SelectU(state, hit, node.children[i], ConstantU32(state, InvalidNode));
		key[i]          = slab.min_t;
	}

	const auto sort_enabled = Binary(state, spv::OpINotEqual, TypeBool(state),
	                                 Binary(state, spv::OpBitwiseAnd, TypeU32(state), desc1,
	                                        ConstantU32(state, DescBoxSortEnable)),
	                                 ConstantU32(state, 0));
	const auto invalid      = ConstantU32(state, InvalidNode);
	const auto sort         = [&](uint32_t a, uint32_t b) {
		// Swap when B is a nearer hit than A, or A missed (misses sink to the end).
		const auto b_valid = Binary(state, spv::OpINotEqual, TypeBool(state), traverse[b], invalid);
		const auto a_missed = Binary(state, spv::OpIEqual, TypeBool(state), traverse[a], invalid);
		const auto nearer   = FCmp(state, spv::OpFOrdLessThan, key[b], key[a]);
		const auto swap =
		    BoolAnd(state, sort_enabled, BoolOr(state, BoolAnd(state, b_valid, nearer), a_missed));
		const auto new_a_node = SelectU(state, swap, traverse[b], traverse[a]);
		const auto new_b_node = SelectU(state, swap, traverse[a], traverse[b]);
		const auto new_a_key  = SelectF(state, swap, key[b], key[a]);
		const auto new_b_key  = SelectF(state, swap, key[a], key[b]);
		traverse[a]           = new_a_node;
		traverse[b]           = new_b_node;
		key[a]                = new_a_key;
		key[b]                = new_b_key;
	};
	sort(0, 2);
	sort(1, 3);
	sort(0, 1);
	sort(2, 3);
	sort(1, 2);
	return {traverse[0], traverse[1], traverse[2], traverse[3]};
}

std::vector<uint32_t> IntersectTriangleNode(EmitterState& state, uint32_t node_addr,
                                            uint32_t node_type, Vec3 origin, Vec3 dir,
                                            uint32_t desc3) {
	// Vertex slots per node type: 0 -> (v0, v1, v2), 1 -> (v1, v3, v2), 2 -> (v2, v3, v4),
	// 3 -> (v2, v4, v0). Only types 0 and 1 fit a 64-byte node; the others read past the
	// vertex block exactly as the hardware mapping describes.
	constexpr uint32_t mapping[4] = {0x210u, 0x231u, 0x432u, 0x042u};
	const auto         u32        = TypeU32(state);
	Vec3               vertices[3];
	for (uint32_t slot = 0; slot < 3u; slot++) {
		uint32_t offset = ConstantU32(state, ((mapping[0] >> (slot * 4u)) & 0xfu) * 12u);
		for (uint32_t type = 1; type < 4u; type++) {
			const auto is_type =
			    Binary(state, spv::OpIEqual, TypeBool(state), node_type, ConstantU32(state, type));
			offset =
			    SelectU(state, is_type,
			            ConstantU32(state, ((mapping[type] >> (slot * 4u)) & 0xfu) * 12u), offset);
		}
		const auto address = Binary(state, spv::OpIAdd, TypeScalarU64(state), node_addr,
		                            Unary(state, spv::OpUConvert, TypeScalarU64(state), offset));
		vertices[slot]     = LoadVec3At(state, address, 0u);
	}
	const auto triangle_id = LoadU32At(state, node_addr, TriangleIdOffset);

	// GPURT fast_intersect_triangle.
	const auto zero   = ConstantF32Value(state, 0.0f);
	const auto one    = ConstantF32Value(state, 1.0f);
	const auto e1     = VSub(state, vertices[1], vertices[0]);
	const auto e2     = VSub(state, vertices[2], vertices[0]);
	const auto e3     = VSub(state, origin, vertices[0]);
	const auto s1     = VCross(state, dir, e2);
	const auto s2     = VCross(state, e3, e1);
	auto       x      = VDot(state, e2, s2);
	auto       y      = VDot(state, s1, e1);
	const auto z      = VDot(state, e3, s1);
	const auto w      = VDot(state, dir, s2);
	const auto t      = FBin(state, spv::OpFDiv, x, y);
	const auto u      = FBin(state, spv::OpFDiv, z, y);
	const auto v      = FBin(state, spv::OpFDiv, w, y);
	auto       missed = BoolOr(state, FCmp(state, spv::OpFOrdLessThan, u, zero),
	                           FCmp(state, spv::OpFOrdGreaterThan, u, one));
	missed            = BoolOr(state, missed, FCmp(state, spv::OpFOrdLessThan, v, zero));
	missed = BoolOr(state, missed,
	                FCmp(state, spv::OpFOrdGreaterThan, FBin(state, spv::OpFAdd, u, v), one));
	missed = BoolOr(state, missed, FCmp(state, spv::OpFOrdLessThan, t, zero));
	x      = SelectF(state, missed, ConstantF32(state, 0x7f800000u), x);
	y      = SelectF(state, missed, one, y);

	// Barycentric swizzle: triangle_id holds per-type 2-bit source selectors for I and J.
	const auto bary0 = FBin(state, spv::OpFSub, FBin(state, spv::OpFSub, y, z), w);
	const auto shift =
	    Binary(state, spv::OpShiftLeftLogical, u32, node_type, ConstantU32(state, 3));
	const auto pick = [&](uint32_t extra_shift) {
		const auto index =
		    Binary(state, spv::OpBitwiseAnd, u32,
		           Binary(state, spv::OpShiftRightLogical, u32, triangle_id,
		                  Binary(state, spv::OpIAdd, u32, shift, ConstantU32(state, extra_shift))),
		           ConstantU32(state, 3));
		const auto is1 =
		    Binary(state, spv::OpIEqual, TypeBool(state), index, ConstantU32(state, 1));
		const auto is2 =
		    Binary(state, spv::OpIEqual, TypeBool(state), index, ConstantU32(state, 2));
		return SelectF(state, is1, z, SelectF(state, is2, w, bary0));
	};
	const auto i_num = pick(0u);
	const auto j_num = pick(2u);

	// triangle_return_mode 1 returns barycentric numerators; mode 0 returns the triangle_id
	// word and a hit flag instead.
	const auto return_ij = Binary(
	    state, spv::OpINotEqual, TypeBool(state),
	    Binary(state, spv::OpBitwiseAnd, u32, desc3, ConstantU32(state, DescTriangleReturnIJ)),
	    ConstantU32(state, 0));
	const auto hit_status = SelectU(state, missed, ConstantU32(state, 0), ConstantU32(state, 1));
	return {EmitBitcastF32ToU32(state, x), EmitBitcastF32ToU32(state, y),
	        SelectU(state, return_ij, EmitBitcastF32ToU32(state, i_num), triangle_id),
	        SelectU(state, return_ij, EmitBitcastF32ToU32(state, j_num), hit_status)};
}

bool ProgramUsesBvh(const IR::Program& program) {
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() == IR::ValueOpcode::BvhIntersectRay) {
				return true;
			}
		}
	}
	return false;
}

} // namespace

void DefineBvhIntersectRay(EmitterState& state) {
	if (!ProgramUsesBvh(state.program)) {
		return;
	}
	if (state.bda_pointer_function == 0) {
		EXIT("bvh intersection requires the BDA page table (uses_dma)\n");
	}
	const auto u32  = TypeU32(state);
	const auto u64  = TypeScalarU64(state);
	const auto f32  = TypeF32(state);
	const auto vec3 = TypeF32Vector(state, 3);
	const auto ret  = TypeU32Vector(state, 4);
	const auto function_type =
	    state.builder.Type(spv::OpTypeFunction, ret, u64, u32, f32, vec3, vec3, vec3, u32, u32);
	state.bvh_intersect_function = state.builder.AllocateId();
	state.builder.AddName(state.bvh_intersect_function, "bvh_intersect_ray");
	state.builder.AddFunction(spv::OpFunction, ret, state.bvh_intersect_function,
	                          spv::FunctionControlMaskNone, function_type);
	const auto node_addr = state.builder.AllocateId();
	const auto node_type = state.builder.AllocateId();
	const auto extent    = state.builder.AllocateId();
	const auto origin_v  = state.builder.AllocateId();
	const auto dir_v     = state.builder.AllocateId();
	const auto inv_dir_v = state.builder.AllocateId();
	const auto desc1     = state.builder.AllocateId();
	const auto desc3     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpFunctionParameter, u64, node_addr);
	state.builder.AddFunction(spv::OpFunctionParameter, u32, node_type);
	state.builder.AddFunction(spv::OpFunctionParameter, f32, extent);
	state.builder.AddFunction(spv::OpFunctionParameter, vec3, origin_v);
	state.builder.AddFunction(spv::OpFunctionParameter, vec3, dir_v);
	state.builder.AddFunction(spv::OpFunctionParameter, vec3, inv_dir_v);
	state.builder.AddFunction(spv::OpFunctionParameter, u32, desc1);
	state.builder.AddFunction(spv::OpFunctionParameter, u32, desc3);
	EmitLabel(state, state.builder.AllocateId());

	const auto unpack = [&](uint32_t vector) {
		return Vec3 {Extract(state, f32, vector, 0), Extract(state, f32, vector, 1),
		             Extract(state, f32, vector, 2)};
	};
	const auto origin  = unpack(origin_v);
	const auto dir     = unpack(dir_v);
	const auto inv_dir = unpack(inv_dir_v);

	const std::vector<uint32_t> types(4u, u32);
	const auto                  invalid_result = [&] {
		const auto invalid = ConstantU32(state, InvalidNode);
		return std::vector<uint32_t> {invalid, invalid, invalid, invalid};
	};
	// One page-table translation per node; an unmapped node reads as a user node (all ones),
	// which the traversal loops treat as a miss.
	const auto host = GetBdaPointer(state, node_addr);
	const auto present =
	    Binary(state, spv::OpINotEqual, TypeBool(state), host, ConstantDeviceAddress(state, 0));
	const auto is_box = BoolOr(state,
	                           Binary(state, spv::OpIEqual, TypeBool(state), node_type,
	                                  ConstantU32(state, BvhNodeTypeBox16)),
	                           Binary(state, spv::OpIEqual, TypeBool(state), node_type,
	                                  ConstantU32(state, BvhNodeTypeBox32)));
	const auto is_triangle =
	    Binary(state, spv::OpULessThan, TypeBool(state), node_type, ConstantU32(state, 4));
	const auto result = EmitIfElseValues(
	    state, present, types,
	    [&] {
		    return EmitIfElseValues(
		        state, is_box, types,
		        [&] {
			        return IntersectBoxNode(state, host, node_type, extent, origin, inv_dir, desc1);
		        },
		        [&] {
			        return EmitIfElseValues(
			            state, is_triangle, types,
			            [&] {
				            return IntersectTriangleNode(state, host, node_type, origin, dir,
				                                         desc3);
			            },
			            invalid_result);
		        });
	    },
	    invalid_result);
	const auto composite = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, ret, composite, result[0], result[1],
	                          result[2], result[3]);
	state.builder.AddFunction(spv::OpReturnValue, composite);
	state.builder.AddFunction(spv::OpFunctionEnd);
}

uint32_t EmitBvhIntersectRay(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&       state = ctx.state;
	const auto& mem   = ctx.Memory(inst);
	if (state.bvh_intersect_function == 0) {
		ctx.Fail(inst, "bvh intersection function was not defined");
	}
	const auto* handle = inst.Arg(0).Resolve().TryInstruction();
	if (handle == nullptr || handle->GetOpcode() != IR::ValueOpcode::GetAddressResource ||
	    handle->NumArgs() != 2) {
		ctx.Fail(inst, "has no BVH base address pair");
	}
	const auto* address = ctx.ImageAddress(inst.Arg(1));
	const bool  a16     = (mem.image_sample_flags & Decoder::ImageSampleFlagA16) != 0u;
	const bool  bvh64   = mem.image_address_components == (a16 ? 9u : 12u);
	const auto  comp    = [&](uint32_t index) { return ctx.Arg(*address, index); };
	const auto  as_f32  = [&](uint32_t index) { return EmitBitcastU32ToF32(state, comp(index)); };

	const auto base    = DeviceAddressFromWords(state, ctx.Arg(*handle, 0), ctx.Arg(*handle, 1));
	const auto node_lo = comp(0);
	const auto node_hi = bvh64 ? comp(1) : ConstantU32(state, 0);
	// node_addr = base + node_pointer[63:3] * 64 bytes; the low three bits hold the node type.
	const auto node64      = DeviceAddressFromWords(state, node_lo, node_hi);
	const auto node_offset = Binary(state, spv::OpShiftLeftLogical, TypeScalarU64(state),
	                                Binary(state, spv::OpBitwiseAnd, TypeScalarU64(state), node64,
	                                       ConstantDeviceAddress(state, ~uint64_t {7})),
	                                ConstantDeviceAddress(state, 3));
	const auto node_addr   = Binary(state, spv::OpIAdd, TypeScalarU64(state), base, node_offset);
	const auto node_type =
	    Binary(state, spv::OpBitwiseAnd, TypeU32(state), node_lo, ConstantU32(state, 7));

	const uint32_t ray    = bvh64 ? 2u : 1u;
	const auto     extent = as_f32(ray);
	Vec3           origin {as_f32(ray + 1u), as_f32(ray + 2u), as_f32(ray + 3u)};
	Vec3           dir;
	Vec3           inv_dir;
	if (a16) {
		// Halves packed as dir.xy, dir.z|inv.x, inv.yz.
		const auto [dx, dy] = UnpackHalves(state, comp(ray + 4u));
		const auto [dz, ix] = UnpackHalves(state, comp(ray + 5u));
		const auto [iy, iz] = UnpackHalves(state, comp(ray + 6u));
		dir                 = {dx, dy, dz};
		inv_dir             = {ix, iy, iz};
	} else {
		dir     = {as_f32(ray + 4u), as_f32(ray + 5u), as_f32(ray + 6u)};
		inv_dir = {as_f32(ray + 7u), as_f32(ray + 8u), as_f32(ray + 9u)};
	}
	const auto vec3 = [&](Vec3 v) {
		const auto id = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 3), id, v.x, v.y,
		                          v.z);
		return id;
	};
	const auto desc1 = ctx.Arg(inst, 2);
	const auto desc3 = ctx.Arg(inst, 4);
	const auto exec  = ctx.Arg(inst, 5);
	return EmitValueOrDefaultIfCondition(
	    state, exec, TypeU32Vector(state, 4), ConstantU32CompositeZero(state, 4), [&] {
		    const auto result = state.builder.AllocateId();
		    state.builder.AddFunction(spv::OpFunctionCall, TypeU32Vector(state, 4), result,
		                              state.bvh_intersect_function, node_addr, node_type, extent,
		                              vec3(origin), vec3(dir), vec3(inv_dir), desc1, desc3);
		    return result;
	    });
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
