using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.Json;

namespace AutoPhotoEditor.Api
{
    public sealed class AutoEnhanceApiResult
    {
        public string JobId { get; init; } = string.Empty;

        public string AnalysisJson { get; init; } = string.Empty;

        public JsonElement MaskReport { get; init; }

        public byte[] ImageBytes { get; init; } = Array.Empty<byte>();
    }
}
