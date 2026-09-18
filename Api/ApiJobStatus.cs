using System.Text.Json.Serialization;

namespace AutoPhotoEditor.Api
{
    public sealed class ApiJobStatus
    {
        [JsonPropertyName("job_id")]
        public string JobId { get; set; } = string.Empty;

        [JsonPropertyName("status")]
        public string? Status { get; set; }

        [JsonPropertyName("stage")]
        public string? Stage { get; set; }

        [JsonPropertyName("progress")]
        public double? Progress { get; set; }

        [JsonPropertyName("message")]
        public string? Message { get; set; }

        [JsonPropertyName("error")]
        public string? Error { get; set; }

        [JsonPropertyName("result")]
        public object? Result { get; set; }

        [JsonPropertyName("latest")]
        public ApiJobEvent? Latest { get; set; }
    }
}
