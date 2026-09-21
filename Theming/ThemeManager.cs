using System;
using System.Windows;
using System.Windows.Media;
using Wpf.Ui.Appearance;

namespace AutoPhotoEditor.Theming;

public static class ThemeManager
{
    private const string SystemMode = "System";

    public static void Apply(Window window, string mode)
    {
        string themeName = IsDark(mode) ? "Dark" : "Light";
        var theme = new ResourceDictionary
        {
            Source = new Uri($"Themes/{themeName}.xaml", UriKind.Relative)
        };

        foreach (object key in theme.Keys)
        {
            if (theme[key] is not SolidColorBrush sourceBrush ||
                window.Resources[key] is not SolidColorBrush targetBrush)
            {
                continue;
            }

            targetBrush.Color = sourceBrush.Color;
        }
    }

    public static bool IsDark(string mode)
    {
        if (string.Equals(mode, "Dark", StringComparison.OrdinalIgnoreCase))
            return true;

        if (string.Equals(mode, "Light", StringComparison.OrdinalIgnoreCase))
            return false;

        SystemThemeManager.UpdateSystemThemeCache();
        return SystemThemeManager.GetCachedSystemTheme() == SystemTheme.Dark;
    }

    public static string NextMode(string mode)
    {
        return mode switch
        {
            "System" => "Light",
            "Light" => "Dark",
            _ => SystemMode
        };
    }

    public static string DisplayName(string mode)
    {
        return mode switch
        {
            "Light" => "Light",
            "Dark" => "Dark",
            _ => "System"
        };
    }

}
