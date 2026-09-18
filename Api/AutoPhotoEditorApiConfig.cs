using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AutoPhotoEditor.Api
{
    public sealed class AutoPhotoEditorApiConfig
    {
        public string Ip { get; set; } = "127.0.0.1";

        public int Port { get; set; } = 8000;

        public string BaseUrl => $"http://{Ip}:{Port}/";
    }
}