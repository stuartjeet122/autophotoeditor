using System;

namespace AutoPhotoEditor.Api
{
    public sealed class AutoPhotoEditorApiConfig
    {
        public string Ip { get; set; } = "127.0.0.1";

        public int Port { get; set; } = 8000;

        public string BaseUrl => $"http://{Ip}:{Port}/";

        public Uri WebSocketUri
        {
            get
            {
                var builder = new UriBuilder(BaseUrl)
                {
                    Scheme = "ws",
                    Path = "/ws"
                };

                return builder.Uri;
            }
        }
    }
}