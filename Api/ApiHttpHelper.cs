using System;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace AutoPhotoEditor.Api;

public sealed class ApiHttpHelper
{
    private readonly HttpClient _client;

    public ApiHttpHelper(
        AutoPhotoEditorApiConfig config)
    {
        if (config == null)
            throw new ArgumentNullException(
                nameof(config));

        _client = new HttpClient
        {
            BaseAddress =
                new Uri(config.BaseUrl),

            Timeout =
                Timeout.InfiniteTimeSpan
        };
    }

    public async Task<string> GetAsync(
        string endpoint,
        CancellationToken cancellationToken = default)
    {
        using HttpResponseMessage response =
            await _client.GetAsync(
                endpoint,
                cancellationToken);

        return await ReadResponseAsync(
            response,
            cancellationToken);
    }

    public async Task<string> PostBytesAsync(
        string endpoint,
        byte[] data,
        string mediaType,
        CancellationToken cancellationToken = default)
    {
        using var content =
            new ByteArrayContent(data);

        content.Headers.ContentType =
            new System.Net.Http.Headers.MediaTypeHeaderValue(
                mediaType);

        using HttpResponseMessage response =
            await _client.PostAsync(
                endpoint,
                content,
                cancellationToken);

        return await ReadResponseAsync(
            response,
            cancellationToken);
    }

    public async Task<string> PostJsonAsync(
        string endpoint,
        object body,
        CancellationToken cancellationToken = default)
    {
        string json =
            System.Text.Json.JsonSerializer.Serialize(body);

        using var content =
            new StringContent(
                json,
                Encoding.UTF8,
                "application/json");

        using HttpResponseMessage response =
            await _client.PostAsync(
                endpoint,
                content,
                cancellationToken);

        return await ReadResponseAsync(
            response,
            cancellationToken);
    }

    private static async Task<string> ReadResponseAsync(
        HttpResponseMessage response,
        CancellationToken cancellationToken)
    {
        string content =
            await response.Content.ReadAsStringAsync(
                cancellationToken);

        if (!response.IsSuccessStatusCode)
        {
            throw new AutoPhotoEditorApiException(
                $"API request failed. " +
                $"HTTP {(int)response.StatusCode} " +
                $"{response.ReasonPhrase}\n\n" +
                content);
        }

        return content;
    }
}