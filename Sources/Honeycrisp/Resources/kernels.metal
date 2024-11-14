#include <metal_stdlib>
using namespace metal;

inline float safe_tanh(float x) {
    return (x < -10 ? -1 : (x > 10 ? 1 : tanh(x)));
}

template <typename T>
inline T pythonFmod(T lhs, T rhs) {
    if (rhs < 0) {
        return -pythonFmod(-lhs, -rhs);
    } else if (lhs < 0) {
        return fmod(rhs - fmod(rhs - lhs, rhs), rhs);
    } else {
        return fmod(lhs, rhs);
    }
}

template <typename T>
inline T pythonIntMod(T lhs, T rhs) {
    if (rhs < 0) {
      return -pythonIntMod(-lhs, -rhs);
    } else if (lhs < 0) {
      return (rhs - ((rhs - lhs) % rhs)) % rhs;
    } else {
      return lhs % rhs;
    }
}

#define BINARY_KERNELS(name, expr, type, outType) \
    struct name##vv_args_##type { \
        uint aCount; \
        uint aDiv; \
        uint bCount; \
        uint bDiv; \
        uint N; \
    }; \
    kernel void name##vv_##type( \
        device const type* a [[buffer(0)]], \
        device const type* b [[buffer(1)]], \
        device outType* c [[buffer(2)]], \
        constant struct name##vv_args_##type &args [[buffer(3)]], \
        uint id [[thread_position_in_grid]] \
    ) { \
        if (id < args.N) { \
            type x = a[(id / args.aDiv) % args.aCount]; \
            type y = b[(id / args.bDiv) % args.bCount]; \
            c[id] = expr; \
        } \
    } \
    kernel void name##vs_##type( \
        device const type* a [[buffer(0)]], \
        device const float& b [[buffer(1)]], \
        device outType* c [[buffer(2)]], \
        constant uint &N [[buffer(3)]], \
        uint id [[thread_position_in_grid]] \
    ) { \
        if (id < N) { \
            type x = a[id]; \
            type y = (type)b; \
            c[id] = expr; \
        } \
    } \
    kernel void name##sv_##type( \
        device const float& a [[buffer(0)]], \
        device const type* b [[buffer(1)]], \
        device outType* c [[buffer(2)]], \
        constant uint &N [[buffer(3)]], \
        uint id [[thread_position_in_grid]] \
    ) { \
        if (id < N) { \
            type x = (type)a; \
            type y = b[id]; \
            c[id] = expr; \
        } \
    }

#define ALL_BINARY_KERNELS_FOR_IN_TYPE(type) \
    BINARY_KERNELS(add, x+y, type, type) \
    BINARY_KERNELS(sub, x-y, type, type) \
    BINARY_KERNELS(mul, x*y, type, type) \
    BINARY_KERNELS(div, x/y, type, type) \
    BINARY_KERNELS(lt, x<y, type, char) \
    BINARY_KERNELS(gt, x>y, type, char) \
    BINARY_KERNELS(le, x<=y, type, char) \
    BINARY_KERNELS(ge, x>=y, type, char) \
    BINARY_KERNELS(eq, x==y, type, char)

ALL_BINARY_KERNELS_FOR_IN_TYPE(half)
ALL_BINARY_KERNELS_FOR_IN_TYPE(float)
ALL_BINARY_KERNELS_FOR_IN_TYPE(long)
BINARY_KERNELS(mod, (pythonFmod(x,y)), half, half)
BINARY_KERNELS(mod, (pythonFmod(x,y)), float, float)
BINARY_KERNELS(mod, (pythonIntMod(x,y)), long, long)

#define BITWISE_KERNELS(name, expr, type) \
    struct name##_args_##type { \
        uint aCount; \
        uint aDiv; \
        uint bCount; \
        uint bDiv; \
        uint N; \
    }; \
    kernel void name##_##type( \
        device const type* a [[buffer(0)]], \
        device const type* b [[buffer(1)]], \
        device type* c [[buffer(2)]], \
        constant struct name##_args_##type &args [[buffer(3)]], \
        uint id [[thread_position_in_grid]] \
    ) { \
        if (id < args.N) { \
            type x = a[(id / args.aDiv) % args.aCount]; \
            type y = b[(id / args.bDiv) % args.bCount]; \
            c[id] = expr; \
        } \
    }

#define ALL_BITWISE_KERNELS(type) \
    BITWISE_KERNELS(xor, x^y, type) \
    BITWISE_KERNELS(or, x|y, type) \
    BITWISE_KERNELS(and, x&y, type)

ALL_BITWISE_KERNELS(char)
ALL_BITWISE_KERNELS(short)
ALL_BITWISE_KERNELS(int)
ALL_BITWISE_KERNELS(long)

#define DEFINE_FUSED_ADD_MUL(type) \
    kernel void add_mul_##type( \
        device const type* a [[buffer(0)]], \
        device const type* b [[buffer(1)]], \
        device const type* c [[buffer(2)]], \
        device type* output [[buffer(3)]], \
        constant uint &aCount [[buffer(4)]], \
        constant uint &aDiv [[buffer(5)]], \
        constant uint &bCount [[buffer(6)]], \
        constant uint &bDiv [[buffer(7)]], \
        constant uint &cCount [[buffer(8)]], \
        constant uint &cDiv [[buffer(9)]], \
        constant uint &N [[buffer(10)]], \
        uint id [[thread_position_in_grid]] \
    ) { \
        if (id < N) { \
            output[id] = (a[(id / aDiv) % aCount] + b[(id / bDiv) % bCount]) * c[(id / cDiv) % cCount]; \
        } \
    } \
    kernel void mul_add_##type( \
        device const type* a [[buffer(0)]], \
        device const type* b [[buffer(1)]], \
        device const type* c [[buffer(2)]], \
        device type* output [[buffer(3)]], \
        constant uint &aCount [[buffer(4)]], \
        constant uint &aDiv [[buffer(5)]], \
        constant uint &bCount [[buffer(6)]], \
        constant uint &bDiv [[buffer(7)]], \
        constant uint &cCount [[buffer(8)]], \
        constant uint &cDiv [[buffer(9)]], \
        constant uint &N [[buffer(10)]], \
        uint id [[thread_position_in_grid]] \
    ) { \
        if (id < N) { \
            output[id] = (a[(id / aDiv) % aCount] * b[(id / bDiv) % bCount]) + c[(id / cDiv) % cCount]; \
        } \
    }

DEFINE_FUSED_ADD_MUL(long)
DEFINE_FUSED_ADD_MUL(float)
DEFINE_FUSED_ADD_MUL(half)

#define DEFINE_NORMALIZE(type) \
    struct normalize_args_##type { \
        float epsilon; \
        uint inputCount; \
        uint inputDiv; \
        uint meanCount; \
        uint meanDiv; \
        uint varianceCount; \
        uint varianceDiv; \
        uint N; \
    }; \
    struct normalize_x_grad_args_##type { \
        float epsilon; \
        float sign; \
        uint varianceCount; \
        uint varianceDiv; \
        uint outGradCount; \
        uint outGradDiv; \
        uint N; \
    }; \
    struct normalize_var_grad_args_##type { \
        float epsilon; \
        uint inputCount; \
        uint inputDiv; \
        uint meanCount; \
        uint meanDiv; \
        uint varianceCount; \
        uint varianceDiv; \
        uint outGradCount; \
        uint outGradDiv; \
        uint N; \
    }; \
    kernel void normalize_##type( \
        device const type* input [[buffer(0)]], \
        device const type* mean [[buffer(1)]], \
        device const type* variance [[buffer(2)]], \
        device type* output [[buffer(3)]], \
        constant struct normalize_args_##type &args [[buffer(4)]], \
        uint id [[thread_position_in_grid]] \
    ) { \
        if (id < args.N) { \
            type x = (float)input[(id / args.inputDiv) % args.inputCount]; \
            type mu = (float)mean[(id / args.meanDiv) % args.meanCount]; \
            type sig = (float)variance[(id / args.varianceDiv) % args.varianceCount]; \
            output[id] = (type)((x - mu) / sqrt(sig + args.epsilon)); \
        } \
    } \
    kernel void normalize_x_grad_##type( \
        device const type* variance [[buffer(0)]], \
        device const type* outGrad [[buffer(1)]], \
        device type* output [[buffer(2)]], \
        constant struct normalize_x_grad_args_##type &args [[buffer(3)]], \
        uint id [[thread_position_in_grid]] \
    ) { \
        if (id < args.N) { \
            type sig = (float)variance[(id / args.varianceDiv) % args.varianceCount]; \
            type g = (float)outGrad[(id / args.outGradDiv) % args.outGradCount]; \
            output[id] = (type)(args.sign * g / sqrt(sig + args.epsilon)); \
        } \
    } \
    kernel void normalize_var_grad_##type( \
        device const type* input [[buffer(0)]], \
        device const type* mean [[buffer(1)]], \
        device const type* variance [[buffer(2)]], \
        device const type* outGrad [[buffer(3)]], \
        device type* output [[buffer(4)]], \
        constant struct normalize_var_grad_args_##type &args [[buffer(5)]], \
        uint id [[thread_position_in_grid]] \
    ) { \
        if (id < args.N) { \
            type x = (float)input[(id / args.inputDiv) % args.inputCount]; \
            type mu = (float)mean[(id / args.meanDiv) % args.meanCount]; \
            type sig = (float)variance[(id / args.varianceDiv) % args.varianceCount]; \
            type g = (float)outGrad[(id / args.outGradDiv) % args.outGradCount]; \
            output[id] = (type)(-g * 0.5 * (x - mu) * pow(sig + args.epsilon, -1.5)); \
        } \
    }

DEFINE_NORMALIZE(half)
DEFINE_NORMALIZE(float)

#define DEFINE_CLAMP(type, minMaxType) \
    kernel void clamp_##type( \
        device const type* a [[buffer(0)]], \
        device type* output [[buffer(1)]], \
        device minMaxType& min [[buffer(2)]], \
        device minMaxType& max [[buffer(3)]], \
        device uint& hasMin [[buffer(4)]], \
        device uint& hasMax [[buffer(5)]], \
        constant uint &N [[buffer(6)]], \
        uint id [[thread_position_in_grid]] \
    ) { \
        if (id < N) { \
            float input = a[id]; \
            if (hasMin && input < (type)min) { \
                input = (type)min; \
            } \
            if (hasMax && input > (type)max) { \
                input = (type)max; \
            } \
            output[id] = input; \
        } \
    }

DEFINE_CLAMP(float, float)
DEFINE_CLAMP(long, long)
DEFINE_CLAMP(half, float)

#define DEFINE_WHEN(type) \
    struct when_args_##type { \
        uint maskCount; \
        uint maskDiv; \
        uint trueCount; \
        uint trueDiv; \
        uint falseCount; \
        uint falseDiv; \
        uint N; \
    }; \
    kernel void when_##type( \
        device const char* mask [[buffer(0)]], \
        device const type* trueIn [[buffer(1)]], \
        device const type* falseIn [[buffer(2)]], \
        device type* output [[buffer(3)]], \
        device when_args_##type& args [[buffer(4)]], \
        uint id [[thread_position_in_grid]] \
    ) { \
        if (id < args.N) { \
            if (mask[(id / args.maskDiv) % args.maskCount]) { \
                output[id] = trueIn[(id / args.trueDiv) % args.trueCount]; \
            } else { \
                output[id] = falseIn[(id / args.falseDiv) % args.falseCount]; \
            } \
        } \
    }

DEFINE_WHEN(char)
DEFINE_WHEN(short)
DEFINE_WHEN(int)
DEFINE_WHEN(long)

kernel void vector_pow_half(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant float &exponent [[buffer(2)]],
    constant float &outScale [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        output[id] = half(outScale * pow(float(input[id]), exponent));
    }
}

kernel void vector_pow_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float &exponent [[buffer(2)]],
    constant float &outScale [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        output[id] = outScale * pow(input[id], exponent);
    }
}

kernel void log_half(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        output[id] = log(input[id]);
    }
}

kernel void log_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        output[id] = log(input[id]);
    }
}

kernel void recip_half(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        output[id] = 1 / input[id];
    }
}

kernel void recip_float(device const float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        constant uint &N [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    if (id < N) {
        output[id] = 1 / input[id];
    }
}

kernel void exp_half(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        output[id] = exp(input[id]);
    }
}

kernel void exp_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        output[id] = exp(input[id]);
    }
}

kernel void sigmoid_half(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        output[id] = (safe_tanh(input[id] / 2) + 1) / 2;
    }
}

kernel void sigmoid_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        output[id] = (safe_tanh(input[id] / 2) + 1) / 2;
    }
}

kernel void sigmoid_grad_half(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        half s = (safe_tanh(input[id] / 2) + 1) / 2;
        output[id] = s * (1 - s);
  }
}

kernel void sigmoid_grad_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        half s = (safe_tanh(input[id] / 2) + 1) / 2;
        output[id] = s * (1 - s);
  }
}

kernel void gelu_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        float x = input[id];
        output[id] = 0.5 * x * (1.0 + safe_tanh(0.797884561 * (x + 0.044715 * pow(x, 3.0))));
    }
}

kernel void gelu_half(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        float x = float(input[id]);
        output[id] = half(0.5 * x * (1.0 + safe_tanh(0.797884561 * (x + 0.044715 * pow(x, 3.0)))));
    }
}

kernel void gelu_grad_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        float x = input[id];
        float tanhTerm = safe_tanh(0.035677408145115 * pow(x, 3.0) + 0.797884561 * x);
        output[id] = 0.5 * x * (1.0 - pow(tanhTerm, 2.0)) * (0.107032224435345 * pow(x, 2.0) + 0.797884561) + 0.5 * tanhTerm + 0.5;
    }
}

kernel void gelu_grad_half(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        float x = float(input[id]);
        float tanhTerm = safe_tanh(0.035677408145115 * pow(x, 3.0) + 0.797884561 * x);
        output[id] = half(0.5 * x * (1.0 - pow(tanhTerm, 2.0)) * (0.107032224435345 * pow(x, 2.0) + 0.797884561) + 0.5 * tanhTerm + 0.5);
    }
}

kernel void sin_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        output[id] = sin(input[id]);
    }
}

kernel void sin_half(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        output[id] = sin(input[id]);
    }
}

kernel void cos_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        output[id] = cos(input[id]);
    }
}

kernel void cos_half(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        output[id] = cos(input[id]);
    }
}

kernel void minus_sin_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        output[id] = -sin(input[id]);
    }
}


kernel void minus_sin_half(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        output[id] = -sin(input[id]);
    }
}

kernel void relu_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        float x = input[id];
        output[id] = x > 0.0f ? x : 0.0f;
    }
}

kernel void relu_half(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        half x = input[id];
        output[id] = x > half(0.0) ? x : half(0.0);
    }
}

kernel void relu_grad_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        float x = input[id];
        output[id] = x > 0.0f ? 1.0f : 0.0f;
    }
}


kernel void relu_grad_half(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        half x = input[id];
        output[id] = x > half(0.0) ? half(1.0) : half(0.0);
    }
}

kernel void abs_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        output[id] = abs(input[id]);
    }
}


kernel void abs_half(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        output[id] = abs(input[id]);
    }
}

kernel void abs_grad_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        output[id] = input[id] < 0 ? -1 : 1;
    }
}


kernel void abs_grad_half(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint &N [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < N) {
        output[id] = input[id] < 0 ? -1 : 1;
    }
}

#define DEFINE_REPEAT(type) \
    struct repeat_args_##type { \
        uint inner; \
        uint outer; \
        uint reps; \
    }; \
    kernel void repeat_##type( \
        device const type* input [[buffer(0)]], \
        device type* output [[buffer(1)]], \
        constant struct repeat_args_##type &args [[buffer(2)]], \
        uint id [[thread_position_in_grid]] \
    ) { \
        if (id < args.inner * args.outer * args.reps) { \
            uint sourceIdx = (id % args.inner) + (id / (args.inner * args.reps)) * args.inner; \
            output[id] = input[sourceIdx]; \
        } \
    }

DEFINE_REPEAT(char)
DEFINE_REPEAT(short)
DEFINE_REPEAT(int)
DEFINE_REPEAT(long)

#define DEFINE_STRIDED_COPY(type) \
    struct strided_copy_args_##type { \
        uint inner; \
        uint fullInner; \
        uint outer; \
        uint offset; \
    }; \
    kernel void strided_copy_##type( \
        device const type* input [[buffer(0)]], \
        device type* output [[buffer(1)]], \
        constant struct strided_copy_args_##type &args [[buffer(2)]], \
        uint id [[thread_position_in_grid]] \
    ) { \
        if (id < args.inner * args.outer) { \
            uint sourceCol = id % args.inner; \
            uint sourceRow = id / args.inner; \
            output[(sourceCol + args.offset) + sourceRow*args.fullInner] = input[id]; \
        } \
    }

DEFINE_STRIDED_COPY(char)
DEFINE_STRIDED_COPY(short)
DEFINE_STRIDED_COPY(int)
DEFINE_STRIDED_COPY(long)

constant uint PHILOX_ROUND_A = 0xD2511F53;
constant uint PHILOX_ROUND_B = 0xCD9E8D57;
constant uint PHILOX_KEY_A = 0x9E3779B9;
constant uint PHILOX_KEY_B = 0xBB67AE85;
constant constexpr float M_PI = 3.14159265358979323846264338327950288;

inline uint umulhi(uint x, uint y) {
    ulong prod = ((ulong)x) * ((ulong)y);
    return (uint)(prod >> 32);
}

inline void philox(uint seed0, uint seed1, uint offset, thread uint* c) {
    c[0] = offset;
    c[1] = 0;
    c[2] = 0;
    c[3] = 0;
    uint k0 = seed0;
    uint k1 = seed1;
  
    for (int i = 0; i < 10; i++) {
        uint prev_c0 = c[0];
        uint prev_c2 = c[2];
        c[0] = umulhi(PHILOX_ROUND_B, c[2]) ^ c[1] ^ k0;
        c[2] = umulhi(PHILOX_ROUND_A, prev_c0) ^ c[3] ^ k1;
        c[1] = PHILOX_ROUND_B * prev_c2;
        c[3] = PHILOX_ROUND_A * prev_c0;
        k0 = (k0 + PHILOX_KEY_A);
        k1 = (k1 + PHILOX_KEY_B);
    }
}

constant uint64_t MAX_ULONG = 0xFFFFFFFFFFFFFFFF;

ulong next_pow_of_two(ulong x) {
    if (x == 0) {
        return 1;
    }

    if (x >= (MAX_ULONG >> 1)) {
        return MAX_ULONG;
    }

    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}

kernel void rand_long(
    device ulong* output [[buffer(0)]],
    constant uint &seed0 [[buffer(1)]],
    constant uint &seed1 [[buffer(2)]],
    constant uint &offset [[buffer(3)]],
    constant uint &size [[buffer(4)]],
    constant ulong &minVal [[buffer(5)]],
    constant ulong &count [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
    ulong sample = 0;
    uint found = 0;
    uint c[4];

    ulong bound = next_pow_of_two(count);

    for (int i = 0; i < 32; i++) {
        philox(seed0, seed1, offset + i + id*32, c);
        ulong v1 = (((ulong)c[0]) | (((ulong)c[1]) << 32)) % bound;
        ulong v2 = (((ulong)c[2]) | (((ulong)c[3]) << 32)) % bound;
        if (v1 < count && !found) {
            found = 1;
            sample = v1;
        }
        if (v2 < count && !found) {
            found = 1;
            sample = v2;
        }
    }
    if (id < size) {
        output[id] = sample + minVal;
    }
}

kernel void rand_float(
    device float* output [[buffer(0)]],
    constant uint &seed0 [[buffer(1)]],
    constant uint &seed1 [[buffer(2)]],
    constant uint &offset [[buffer(3)]],
    constant uint &size [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    uint c[4];
    philox(seed0, seed1, offset + id, c);
    for (int i = 0; i < 4; i++) {
        uint outIdx = id * 4 + i;
        if (outIdx < size) {
            output[outIdx] = float(c[i]) / float(0xffffffff);
        }
    }
}

kernel void rand_half(
    device half* output [[buffer(0)]],
    constant uint &seed0 [[buffer(1)]],
    constant uint &seed1 [[buffer(2)]],
    constant uint &offset [[buffer(3)]],
    constant uint &size [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    uint c[4];
    philox(seed0, seed1, offset + id, c);
    for (int i = 0; i < 4; i++) {
        uint outIdx = id * 4 + i;
        if (outIdx < size) {
            output[outIdx] = half(float(c[i]) / float(0xffffffff));
        }
    }
}

kernel void randn_float(
    device float* output [[buffer(0)]],
    constant uint &seed0 [[buffer(1)]],
    constant uint &seed1 [[buffer(2)]],
    constant uint &offset [[buffer(3)]],
    constant uint &size [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    uint c[4];
    philox(seed0, seed1, offset + id, c);
    float u1 = float(c[0]) / float(0xffffffff);
    if (u1 < 1e-5) {
        u1 = 1e-5;
    }
    float u2 = float(c[1]) / float(0xffffffff);
    float r = sqrt(-2 * log(u1));
    float phi = 2 * M_PI * u2;
    float z[2];
    z[0] = r * cos(phi);
    z[1] = r * sin(phi);

    for (int i = 0; i < 2; i++) {
        uint outIdx = id * 2 + i;
        if (outIdx < size) {
            output[outIdx] = z[i];
        }
    }
}

kernel void randn_half(
    device half* output [[buffer(0)]],
    constant uint &seed0 [[buffer(1)]],
    constant uint &seed1 [[buffer(2)]],
    constant uint &offset [[buffer(3)]],
    constant uint &size [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    uint c[4];
    philox(seed0, seed1, offset + id, c);
    float u1 = float(c[0]) / float(0xffffffff);
    if (u1 < 1e-5) {
        u1 = 1e-5;
    }
    float u2 = float(c[1]) / float(0xffffffff);
    float r = sqrt(-2 * log(u1));
    float phi = 2 * M_PI * u2;
    float z[2];
    z[0] = r * cos(phi);
    z[1] = r * sin(phi);

    for (int i = 0; i < 2; i++) {
        uint outIdx = id * 2 + i;
        if (outIdx < size) {
            output[outIdx] = half(z[i]);
        }
    }
}

template <typename T>
void gather_bcast_impl(
    device const T* input,
    device const ulong* indices,
    device T* output,
    uint outerCount,
    uint indexCount,
    uint middleCount,
    uint innerCount,
    uint id
) {
    uint innerIdx = id % innerCount;
    uint indexIdx = (id / innerCount) % indexCount;
    uint outerIdx = (id / innerCount) / indexCount;
    if (outerIdx < outerCount) {
        ulong index = indices[indexIdx];
        ulong srcIndex = innerIdx + index*innerCount + outerIdx*innerCount*middleCount;
        output[id] = input[srcIndex];
    }
}

template <typename T>
void gather_impl(
    device const T* input,
    device const ulong* indices,
    device T* output,
    uint outerCount,
    uint indexCount,
    uint middleCount,
    uint innerCount,
    uint id
) {
    uint innerIdx = id % innerCount;
    uint indexIdx = (id / innerCount) % indexCount;
    uint outerIdx = (id / innerCount) / indexCount;
    if (outerIdx < outerCount) {
        ulong index = indices[innerIdx + indexIdx*innerCount + outerIdx*innerCount*indexCount];
        ulong srcIndex = innerIdx + index*innerCount + outerIdx*innerCount*middleCount;
        output[id] = input[srcIndex];
    }
}

#define DEFINE_GATHER(gather, type) \
kernel void gather##_##type( \
    device const type* input [[buffer(0)]], \
    device const ulong* indices [[buffer(1)]], \
    device type* output [[buffer(2)]], \
    constant uint &outerCount [[buffer(3)]], \
    constant uint &indexCount [[buffer(4)]], \
    constant uint &middleCount [[buffer(5)]], \
    constant uint &innerCount [[buffer(6)]], \
    uint id [[thread_position_in_grid]] \
) { \
    gather##_impl<type>(input, indices, output, outerCount, indexCount, middleCount, \
                        innerCount, id); \
}

DEFINE_GATHER(gather, char)
DEFINE_GATHER(gather_bcast, char)
DEFINE_GATHER(gather, short)
DEFINE_GATHER(gather_bcast, short)
DEFINE_GATHER(gather, int)
DEFINE_GATHER(gather_bcast, int)
DEFINE_GATHER(gather, long)
DEFINE_GATHER(gather_bcast, long)

template <typename T>
void scatter_bcast_impl(
    device const T* input,
    device const ulong* indices,
    device T* output,
    uint outerCount,
    uint indexCount,
    uint middleCount,
    uint innerCount,
    uint id
) {
    uint innerIdx = id % innerCount;
    uint indexIdx = (id / innerCount) % indexCount;
    uint outerIdx = (id / innerCount) / indexCount;
    if (outerIdx < outerCount) {
        ulong index = indices[indexIdx];
        ulong dstIndex = innerIdx + index*innerCount + outerIdx*innerCount*middleCount;
        output[dstIndex] = input[id];
    }
}

template <typename T>
void scatter_impl(
    device const T* input,
    device const ulong* indices,
    device T* output,
    uint outerCount,
    uint indexCount,
    uint middleCount,
    uint innerCount,
    uint id
) {
    uint innerIdx = id % innerCount;
    uint indexIdx = (id / innerCount) % indexCount;
    uint outerIdx = (id / innerCount) / indexCount;
    if (outerIdx < outerCount) {
        ulong index = indices[innerIdx + indexIdx*innerCount + outerIdx*innerCount*indexCount];
        ulong dstIndex = innerIdx + index*innerCount + outerIdx*innerCount*middleCount;
        output[dstIndex] = input[id];
    }
}

#define DEFINE_SCATTER(scatter, type) \
    kernel void scatter##_##type( \
        device const type* input [[buffer(0)]], \
        device const ulong* indices [[buffer(1)]], \
        device type* output [[buffer(2)]], \
        constant uint &outerCount [[buffer(3)]], \
        constant uint &indexCount [[buffer(4)]], \
        constant uint &middleCount [[buffer(5)]], \
        constant uint &innerCount [[buffer(6)]], \
        uint id [[thread_position_in_grid]] \
    ) { \
        scatter##_impl<type>(input, indices, output, outerCount, \
                             indexCount, middleCount, innerCount, id); \
    }

DEFINE_SCATTER(scatter, char)
DEFINE_SCATTER(scatter_bcast, char)
DEFINE_SCATTER(scatter, short)
DEFINE_SCATTER(scatter_bcast, short)
DEFINE_SCATTER(scatter, int)
DEFINE_SCATTER(scatter_bcast, int)
DEFINE_SCATTER(scatter, long)
DEFINE_SCATTER(scatter_bcast, long)

kernel void axis_permutation(
    device const uint* newStrides [[buffer(0)]],
    device const uint* newShape [[buffer(1)]],
    device const uint* permutedStrides [[buffer(2)]],
    device ulong* output [[buffer(3)]],
    constant uint &numAxes [[buffer(4)]],
    constant uint &outputCount [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    uint flatIndex = 0;
    for (uint i = 0; i < numAxes; i++) {
        uint oldIdx = (id / newStrides[i]) % newShape[i];
        flatIndex += oldIdx * permutedStrides[i];
    }
    if (id < outputCount) {
        output[id] = (ulong)flatIndex;
    }
}

#define DEFINE_CAST(inType, outType) \
    kernel void cast_##inType##_##outType( \
        device const inType* input [[buffer(0)]], \
        device outType* output [[buffer(1)]], \
        constant uint &count [[buffer(2)]], \
        uint id [[thread_position_in_grid]] \
    ) { \
        if (id < count) { \
            output[id] = (outType)input[id]; \
        } \
    }

DEFINE_CAST(float, half)
DEFINE_CAST(float, long)
DEFINE_CAST(half, float)
DEFINE_CAST(half, long)
DEFINE_CAST(long, float)
DEFINE_CAST(long, half)