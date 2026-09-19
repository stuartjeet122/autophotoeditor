using CommunityToolkit.Mvvm.ComponentModel;

namespace AutoPhotoEditor.Models;

public partial class MaskOption : ObservableObject
{
    public MaskOption(string name, string? binaryPngBase64)
    {
        Name = name;
        BinaryPngBase64 = binaryPngBase64;
    }

    public string Name { get; }

    public string? BinaryPngBase64 { get; }

    [ObservableProperty]
    private bool isSelected;
}
