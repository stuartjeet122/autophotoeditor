namespace AutoPhotoEditor.Models
{
    public sealed class LUTItem
    {
        /// <summary>
        /// Display name shown in the UI.
        /// </summary>
        public string Name { get; set; } = string.Empty;


        /// <summary>
        /// Short description of the LUT.
        /// </summary>
        public string Description { get; set; } = string.Empty;


        /// <summary>
        /// Optional remote download URL.
        /// </summary>
        public string DownloadUrl { get; set; } = string.Empty;


        /// <summary>
        /// Optional preview image URL.
        /// </summary>
        public string PreviewUrl { get; set; } = string.Empty;


        /// <summary>
        /// Local .cube file used by the API.
        /// </summary>
        public string FilePath { get; set; } = string.Empty;


        public override string ToString()
        {
            return Name;
        }
    }
}