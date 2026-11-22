struct Ray
{
    float3 Origin;
    float TMin;

    float3 Direction;
    float TMax;
};

struct RayGenParameters
{
    float4 Row0;
    float4 Row1;
    float4 Row2;

    float RangeMin;
    float RangeMax;
    uint FirstRay;
    uint RayCount;
    uint UseRayIndices;
};

struct LocalDirection
{
    float X;
    float Y;
    float Z;
};

[[vk::binding(0, 0)]]
RWStructuredBuffer<Ray> rays;

[[vk::binding(1, 0)]]
StructuredBuffer<uint> rayIndices;

[[vk::binding(2, 0)]]
StructuredBuffer<LocalDirection> localDirections;

[[vk::push_constant]]
ConstantBuffer<RayGenParameters> parameters;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= parameters.RayCount)
    {
        return;
    }

    uint index = parameters.FirstRay + id.x;
    if (parameters.UseRayIndices != 0u)
    {
        index = rayIndices[id.x];
    }
    LocalDirection local = localDirections[index];
    precise float3 world = float3(
        parameters.Row0.x * local.X + parameters.Row0.y * local.Y + parameters.Row0.z * local.Z,
        parameters.Row1.x * local.X + parameters.Row1.y * local.Y + parameters.Row1.z * local.Z,
        parameters.Row2.x * local.X + parameters.Row2.y * local.Y + parameters.Row2.z * local.Z);
    precise float length = sqrt(world.x * world.x + world.y * world.y + world.z * world.z);
    precise float3 direction = world / length;

    Ray ray;
    ray.Origin = float3(parameters.Row0.w, parameters.Row1.w, parameters.Row2.w);
    ray.TMin = parameters.RangeMin;
    ray.Direction = direction;
    ray.TMax = parameters.RangeMax;
    rays[id.x] = ray;
}
