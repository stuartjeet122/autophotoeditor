using System;
using System.IO;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace AutoPhotoEditor.Api;

public sealed class ApiWebSocketConnection : IAsyncDisposable
{
    private const int HeartbeatIntervalSeconds = 10;
    private const int ReceiveBufferSize = 4096;
    private const int InitialReconnectDelaySeconds = 1;
    private const int MaximumReconnectDelaySeconds = 15;

    private readonly AutoPhotoEditorApiConfig _config;
    private readonly SemaphoreSlim _lifecycleLock = new(1, 1);
    private CancellationTokenSource? _runCts;
    private Task? _runTask;
    private ClientWebSocket? _socket;
    private bool _disposed;
    private DateTimeOffset? _lastHeartbeatUtc;
    private ApiConnectionState _state = ApiConnectionState.Stopped;

    public ApiWebSocketConnection(AutoPhotoEditorApiConfig config)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
    }

    public event EventHandler<ApiConnectionStatusChangedEventArgs>? StatusChanged;

    public ApiConnectionState State => _state;

    public DateTimeOffset? LastHeartbeatUtc => _lastHeartbeatUtc;

    public async Task StartAsync(CancellationToken cancellationToken = default)
    {
        await _lifecycleLock.WaitAsync(cancellationToken);
        try
        {
            ThrowIfDisposed();

            if (_runTask is { IsCompleted: false })
                return;

            _runCts?.Dispose();
            _runCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            _runTask = RunAsync(_runCts.Token);
        }
        finally
        {
            _lifecycleLock.Release();
        }
    }

    public async Task StopAsync()
    {
        Task? runTask;

        await _lifecycleLock.WaitAsync();
        try
        {
            if (_runCts == null)
            {
                SetStatus(ApiConnectionState.Stopped, "API connection stopped.");
                return;
            }

            _runCts.Cancel();
            runTask = _runTask;
        }
        finally
        {
            _lifecycleLock.Release();
        }

        if (runTask != null)
        {
            try
            {
                await runTask;
            }
            catch (OperationCanceledException)
            {
            }
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;

        _disposed = true;
        await StopAsync();
        _runCts?.Dispose();
        _lifecycleLock.Dispose();
    }

    private async Task RunAsync(CancellationToken cancellationToken)
    {
        int reconnectDelaySeconds = InitialReconnectDelaySeconds;
        SetStatus(ApiConnectionState.Connecting, "Connecting to API service…");

        try
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                try
                {
                    await ConnectAndMonitorAsync(cancellationToken);
                    reconnectDelaySeconds = InitialReconnectDelaySeconds;
                }
                catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
                {
                    break;
                }
                catch (WebSocketException)
                {
                    SetStatus(ApiConnectionState.Reconnecting, "API connection lost. Reconnecting…");
                }
                catch (IOException)
                {
                    SetStatus(ApiConnectionState.Reconnecting, "API connection lost. Reconnecting…");
                }
                catch (InvalidOperationException)
                {
                    SetStatus(ApiConnectionState.Reconnecting, "API connection is unavailable. Reconnecting…");
                }

                await Task.Delay(
                    TimeSpan.FromSeconds(reconnectDelaySeconds),
                    cancellationToken);

                reconnectDelaySeconds = Math.Min(
                    reconnectDelaySeconds * 2,
                    MaximumReconnectDelaySeconds);
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
        }
        finally
        {
            await CloseSocketAsync();
            SetStatus(ApiConnectionState.Stopped, "API connection stopped.");
        }
    }

    private async Task ConnectAndMonitorAsync(CancellationToken cancellationToken)
    {
        await CloseSocketAsync();

        var socket = new ClientWebSocket();
        _socket = socket;

        try
        {
            SetStatus(ApiConnectionState.Connecting, "Connecting to API service…");
            await socket.ConnectAsync(_config.WebSocketUri, cancellationToken);

            SetStatus(ApiConnectionState.Connected, "API connected.");

            using PeriodicTimer heartbeatTimer = new(
                TimeSpan.FromSeconds(HeartbeatIntervalSeconds));

            while (await heartbeatTimer.WaitForNextTickAsync(cancellationToken))
            {
                await SendHeartbeatAsync(socket, cancellationToken);
                await ReceivePongAsync(socket, cancellationToken);
            }
        }
        finally
        {
            if (ReferenceEquals(_socket, socket))
                _socket = null;

            socket.Dispose();
        }
    }

    private static async Task SendHeartbeatAsync(
        ClientWebSocket socket,
        CancellationToken cancellationToken)
    {
        string payload = JsonSerializer.Serialize(
            new
            {
                type = "ping",
                timestamp = DateTimeOffset.UtcNow
            });

        byte[] bytes = Encoding.UTF8.GetBytes(payload);
        await socket.SendAsync(
            new ArraySegment<byte>(bytes),
            WebSocketMessageType.Text,
            true,
            cancellationToken);
    }

    private async Task ReceivePongAsync(
        ClientWebSocket socket,
        CancellationToken cancellationToken)
    {
        byte[] buffer = new byte[ReceiveBufferSize];
        using var message = new MemoryStream();

        while (true)
        {
            WebSocketReceiveResult result = await socket.ReceiveAsync(
                new ArraySegment<byte>(buffer),
                cancellationToken);

            if (result.MessageType == WebSocketMessageType.Close)
                throw new WebSocketException("The API closed the WebSocket connection.");

            if (result.MessageType != WebSocketMessageType.Text)
                continue;

            message.Write(buffer, 0, result.Count);
            if (!result.EndOfMessage)
                continue;

            message.Position = 0;
            using JsonDocument document = await JsonDocument.ParseAsync(
                message,
                cancellationToken: cancellationToken);

            if (document.RootElement.TryGetProperty("type", out JsonElement type) &&
                string.Equals(type.GetString(), "pong", StringComparison.OrdinalIgnoreCase))
            {
                _lastHeartbeatUtc = DateTimeOffset.UtcNow;
                SetStatus(ApiConnectionState.Connected, "API connected.");
                return;
            }

            message.SetLength(0);
        }
    }

    private async Task CloseSocketAsync()
    {
        ClientWebSocket? socket = Interlocked.Exchange(ref _socket, null);
        if (socket == null)
            return;

        try
        {
            if (socket.State is WebSocketState.Open or WebSocketState.CloseReceived)
            {
                using var closeCts = new CancellationTokenSource(TimeSpan.FromSeconds(2));
                await socket.CloseAsync(
                    WebSocketCloseStatus.NormalClosure,
                    "Client shutting down",
                    closeCts.Token);
            }
        }
        catch (OperationCanceledException)
        {
        }
        catch (WebSocketException)
        {
        }
        finally
        {
            socket.Dispose();
        }
    }

    private void SetStatus(ApiConnectionState state, string message)
    {
        _state = state;
        StatusChanged?.Invoke(
            this,
            new ApiConnectionStatusChangedEventArgs(
                state,
                message,
                _lastHeartbeatUtc));
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(ApiWebSocketConnection));
    }
}
