// Generates HDL-64E style rays on the GPU, matching simulation::LidarSensor ray for ray.
// Ray index = beam * AzimuthCount + azimuth.
struct Ray
{
    float3 Origin;
    float TMin;

    float3 Direction;
    float TMax;
};

struct RayGenParameters
{
    // Rows of the 3x4 sensor-to-world transform [R | t], in the local frame.
    float4 Row0;
    float4 Row1;
    float4 Row2;

    uint BeamCount;
    uint AzimuthCount;
    float ElevationMin;
    float ElevationMax;

    float RangeMin;
    float RangeMax;
    uint Padding0;
    uint Padding1;
};

[[vk::binding(0, 0)]]
RWStructuredBuffer<Ray> rays;

[[vk::push_constant]]
ConstantBuffer<RayGenParameters> parameters;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint index = id.x;

    if (index >= parameters.BeamCount * parameters.AzimuthCount)
    {
        return;
    }

    uint beam = index / parameters.AzimuthCount;
    uint step = index % parameters.AzimuthCount;

    float t = parameters.BeamCount > 1 ? float(beam) / float(parameters.BeamCount - 1) : 0.0f;
    float elevation = parameters.ElevationMin + (parameters.ElevationMax - parameters.ElevationMin) * t;
    float azimuth = 6.28318530718f * float(step) / float(parameters.AzimuthCount);

    float3 local = float3(cos(elevation) * cos(azimuth), cos(elevation) * sin(azimuth), sin(elevation));
    float3 world = float3(dot(parameters.Row0.xyz, local), dot(parameters.Row1.xyz, local),
                          dot(parameters.Row2.xyz, local));

    Ray ray;
    ray.Origin = float3(parameters.Row0.w, parameters.Row1.w, parameters.Row2.w);
    ray.TMin = parameters.RangeMin;
    ray.Direction = normalize(world);
    ray.TMax = parameters.RangeMax;

    rays[index] = ray;
}
