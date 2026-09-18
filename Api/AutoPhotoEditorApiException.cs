using System;

namespace AutoPhotoEditor.Api;

public sealed class AutoPhotoEditorApiException
    : Exception
{
    public AutoPhotoEditorApiException(
        string message)
        : base(message)
    {
    }

    public AutoPhotoEditorApiException(
        string message,
        Exception innerException)
        : base(message, innerException)
    {
    }
}