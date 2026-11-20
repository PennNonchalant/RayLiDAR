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
    float Alpha;
    float HardwareT;
};

bool IsBefore(float leftT, uint leftIndex, float rightT, uint rightIndex)
{
    return leftT < rightT || (leftT == rightT && leftIndex < rightIndex);
}

float ResumeTMin(float currentMin, float hardwareT)
{
    return hardwareT > currentMin ? max(currentMin, asfloat(asuint(hardwareT) - 1u)) : currentMin;
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
        float nextHardwareT = asfloat(0x7f800000u);
        float returnUpperBound = asfloat(0x7f800000u);

        RayQuery<RAY_FLAG_NONE> query;
        query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, desc);
        while (query.Proceed())
        {
            uint gaussianIndex = query.CandidateRayInstanceCustomIndex() * parameters.GaussiansPerBatch +
                                 query.CandidateRayPrimitiveIndex() / 2u;
            float hardwareT = query.CandidateTriangleRayT();
            if (hardwareT > returnUpperBound)
            {
                query.CommitNonOpaqueTriangleHit();
            }
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
                if (IsBefore(farthest.T, farthest.GaussianIndex, t, gaussianIndex))
                {
                    nextHardwareT = min(nextHardwareT, hardwareT);
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

            float alpha = gaussian.Opacity * exp(-0.5f * radiusSquared);
            if (accumulatedAlpha + (1.0f - accumulatedAlpha) * alpha >= parameters.AlphaThreshold)
            {
                returnUpperBound = min(returnUpperBound, max(t, hardwareT));
            }

            uint insertIndex = 0;
            bool duplicate = false;
            for (; insertIndex < candidateCount; ++insertIndex)
            {
                GaussianCandidate candidate = candidates[insertIndex];
                if (candidate.GaussianIndex == gaussianIndex)
                {
                    candidates[insertIndex].HardwareT = min(candidate.HardwareT, hardwareT);
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

            if (candidateCount == kHitBatchSize)
            {
                nextHardwareT = min(nextHardwareT, candidates[kHitBatchSize - 1].HardwareT);
            }
            candidateCount = min(candidateCount + 1u, kHitBatchSize);
            for (uint slot = candidateCount - 1; slot > insertIndex; --slot)
            {
                candidates[slot] = candidates[slot - 1];
            }

            GaussianCandidate candidate;
            candidate.T = t;
            candidate.GaussianIndex = gaussianIndex;
            candidate.Alpha = alpha;
            candidate.HardwareT = hardwareT;
            candidates[insertIndex] = candidate;
        }

        for (uint slot = 0; slot < candidateCount; ++slot)
        {
            GaussianCandidate candidate = candidates[slot];
            accumulatedAlpha += (1.0f - accumulatedAlpha) * candidate.Alpha;
            if (accumulatedAlpha >= parameters.AlphaThreshold)
            {
                hit.T = candidate.T;
                hit.Valid = 1;
                hits[index] = hit;
                return;
            }
        }

        if (isinf(nextHardwareT))
        {
            break;
        }

        GaussianCandidate last = candidates[candidateCount - 1];
        hasPrevious = true;
        previousT = last.T;
        previousIndex = last.GaussianIndex;
        desc.TMin = ResumeTMin(desc.TMin, nextHardwareT);
    }

    hits[index] = hit;
}
