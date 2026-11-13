#ifndef RAYLIDAR_GAUSSIAN_BATCH_SIZE
#define RAYLIDAR_GAUSSIAN_BATCH_SIZE 1
#endif

static const uint kHitBatchSize = RAYLIDAR_GAUSSIAN_BATCH_SIZE;

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

struct GaussianCandidate
{
    float T;
    uint GaussianIndex;
    float RadiusSquared;
    float Opacity;
};

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
        GaussianCandidate candidates[kHitBatchSize];
        uint candidateCount = 0;

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
            if (candidateCount == kHitBatchSize)
            {
                GaussianCandidate farthest = candidates[kHitBatchSize - 1];
                if (!IsBefore(t, gaussianIndex, farthest.T, farthest.GaussianIndex))
                {
                    continue;
                }
            }

            float3 delta = offset + t * ray.Direction;
            float2 uv = float2(dot(delta, gaussian.AxisU) / gaussian.SigmaU,
                           dot(delta, gaussian.AxisV) / gaussian.SigmaV);
            float radiusSquared = dot(uv, uv);
            if (radiusSquared > cutoffSquared)
            {
                continue;
            }

            uint insertIndex = 0;
            bool duplicate = false;
            for (; insertIndex < candidateCount; ++insertIndex)
            {
                GaussianCandidate candidate = candidates[insertIndex];
                if (candidate.GaussianIndex == gaussianIndex)
                {
                    duplicate = true;
                    break;
                }
                if (IsBefore(t, gaussianIndex, candidate.T, candidate.GaussianIndex))
                {
                    break;
                }
            }
            if (duplicate)
            {
                continue;
            }

            candidateCount = min(candidateCount + 1u, kHitBatchSize);
            for (uint slot = candidateCount - 1; slot > insertIndex; --slot)
            {
                candidates[slot] = candidates[slot - 1];
            }

            GaussianCandidate candidate;
            candidate.T = t;
            candidate.GaussianIndex = gaussianIndex;
            candidate.RadiusSquared = radiusSquared;
            candidate.Opacity = gaussian.Opacity;
            candidates[insertIndex] = candidate;
        }

        for (uint slot = 0; slot < candidateCount; ++slot)
        {
            GaussianCandidate candidate = candidates[slot];
            float alpha = candidate.Opacity * exp(-0.5f * candidate.RadiusSquared);
            accumulatedAlpha += (1.0f - accumulatedAlpha) * alpha;
            if (accumulatedAlpha >= parameters.AlphaThreshold)
            {
                hit.T = candidate.T;
                hit.Valid = 1;
                hits[index] = hit;
                return;
            }
        }

        if (candidateCount < kHitBatchSize)
        {
            break;
        }

        GaussianCandidate last = candidates[candidateCount - 1];
        hasPrevious = true;
        previousT = last.T;
        previousIndex = last.GaussianIndex;
    }

    hits[index] = hit;
}