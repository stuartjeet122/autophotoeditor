using System;

namespace AutoPhotoEditor.Api;

public sealed class ApiConnectionStatusChangedEventArgs : EventArgs
{
    public ApiConnectionStatusChangedEventArgs(
        ApiConnectionState state,
        string message,
        DateTimeOffset? lastHeartbeatUtc)
    {
        State = state;
        Message = message;
        LastHeartbeatUtc = lastHeartbeatUtc;
    }

    public ApiConnectionState State { get; }

    public string Message { get; }

    public DateTimeOffset? LastHeartbeatUtc { get; }
}
