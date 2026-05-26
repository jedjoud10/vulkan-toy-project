#language slang 2026

struct UniformThing {
    float4x4 view_matrix;
    float4x4 projection_matrix;
    float4x4 inv_view_matrix;
    float4x4 inv_projection_matrix;
    float4x4 view_projection_matrix;
    float2 screen_resolution;
    float2 padding;
    float4 position;
    float4 sun;
    uint32_t debug_type;
    float32_t time;
};