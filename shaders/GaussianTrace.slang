struct Ray
{
    float3 Origin;
    float TMin;

    float3 Direction;
    float TMax;
};

struct Hit
{
    float T;
    uint Valid;
    uint InstanceId;
    uint PrimitiveId;
    float2 Barycentrics;
};

struct Gaussian
{
    float3 Center;
    float Opacity;
    float3 AxisU;
    float SigmaU;
    float3 AxisV;
    float SigmaV;
};

struct GaussianTraceParameters
{
    uint RayCount;
    uint GaussiansPerBatch;
    float SigmaCutoff;
    float AlphaThreshold;
};

[[vk::binding(0, 0)]]
RaytracingAccelerationStructure scene;

[[vk::binding(1, 0)]]
StructuredBuffer<Ray> rays;

[[vk::binding(2, 0)]]
RWStructuredBuffer<Hit> hits;

[[vk::binding(3, 0)]]
StructuredBuffer<Gaussian> gaussians;

[[vk::push_constant]]
ConstantBuffer<GaussianTraceParameters> parameters;

bool IsBefore(float leftT, uint leftIndex, float rightT, uint rightIndex)
{
    return leftT < rightT || (leftT == rightT && leftIndex < rightIndex);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint index = id.x;
    if (index >= parameters.RayCount)
    {
        return;
    }

    Ray ray = rays[index];
    RayDesc desc;
    desc.Origin = ray.Origin;
    desc.Direction = ray.Direction;
    desc.TMin = ray.TMin;
    desc.TMax = ray.TMax;

    Hit hit;
    hit.T = ray.TMax;
    hit.Valid = 0;
    hit.InstanceId = 0xFFFFFFFFu;
    hit.PrimitiveId = 0xFFFFFFFFu;
    hit.Barycentrics = float2(0.0f, 0.0f);

    float cutoffSquared = parameters.SigmaCutoff * parameters.SigmaCutoff;
    float accumulatedAlpha = 0.0f;
    bool hasPrevious = false;
    float previousT = 0.0f;
    uint previousIndex = 0;

    while (true)
    {
        bool found = false;
        float nearestT = 0.0f;
        uint nearestIndex = 0;
        float nearestRadiusSquared = 0.0f;
        float nearestOpacity = 0.0f;

        RayQuery<RAY_FLAG_NONE> query;
        query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, desc);
        while (query.Proceed())
        {
            uint gaussianIndex = query.CandidateRayInstanceCustomIndex() * parameters.GaussiansPerBatch +
                                 query.CandidateRayPrimitiveIndex() / 2u;
            Gaussian gaussian = gaussians[gaussianIndex];
            float3 normal = cross(gaussian.AxisU, gaussian.AxisV);
            float denominator = dot(normal, ray.Direction);
            if (denominator == 0.0f)
            {
                continue;
            }

            float3 offset = ray.Origin - gaussian.Center;
            float t = -dot(normal, offset) / denominator;
            if (t < ray.TMin || t > ray.TMax)
            {
                continue;
            }
            if (hasPrevious && !IsBefore(previousT, previousIndex, t, gaussianIndex))
            {
                continue;
            }
            if (found && !IsBefore(t, gaussianIndex, nearestT, nearestIndex))
            {
                continue;
            }

            float3 delta = offset + t * ray.Direction;
            float2 uv = float2(dot(delta, gaussian.AxisU) / gaussian.SigmaU,
                               dot(delta, gaussian.AxisV) / gaussian.SigmaV);
            float radiusSquared = dot(uv, uv);
            if (radiusSquared > cutoffSquared)
            {
                continue;
            }

            found = true;
            nearestT = t;
            nearestIndex = gaussianIndex;
            nearestRadiusSquared = radiusSquared;
            nearestOpacity = gaussian.Opacity;
        }

        if (!found)
        {
            break;
        }

        float alpha = nearestOpacity * exp(-0.5f * nearestRadiusSquared);
        accumulatedAlpha += (1.0f - accumulatedAlpha) * alpha;
        if (accumulatedAlpha >= parameters.AlphaThreshold)
        {
            hit.T = nearestT;
            hit.Valid = 1;
            break;
        }

        hasPrevious = true;
        previousT = nearestT;
        previousIndex = nearestIndex;
    }

    hits[index] = hit;
}