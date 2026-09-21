using AutoPhotoEditor.Api;
using AutoPhotoEditor.Models;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace AutoPhotoEditor.ViewModels;

public partial class MainViewModel : ObservableObject
{
    private readonly AutoPhotoEditorApiClient _api;
    private readonly ApiWebSocketConnection _apiConnection;

    private CancellationTokenSource? _processingCts;

    private CancellationTokenSource? _imageDataPreloadCts;

    private string? _maskJsonForOriginal;

    private string? _analysisJsonForOriginal;

    private string? _maskImagePath;

    private string? _analysisImagePath;

    private string? _activeJobId;

    // ================================================================
    // DISPLAY CONFIGURATION
    // ================================================================

    private const int DisplayPreviewMaxDimension = 1600;

    public const int PreviewMaxDimension = 1000;

    public const int PreviewJpegQuality = 82;


    // ================================================================
    // ORIGINAL IMAGE
    // ================================================================

    // IMPORTANT:
    // These always refer to the untouched image selected by the user.

    [ObservableProperty]
    private BitmapImage? originalImage;

    [ObservableProperty]
    private string? originalImagePath;


    // ================================================================
    // CURRENT EDITED IMAGE
    // ================================================================

    // IMPORTANT:
    // These always refer to the currently selected edited result.

    [ObservableProperty]
    private BitmapImage? editedImage;

    [ObservableProperty]
    private string? editedImagePath;


    // ================================================================
    // LUT
    // ================================================================

    [ObservableProperty]
    private LUTItem? selectedLut;

    public ObservableCollection<LUTItem> LUTs { get; } = new();


    // ================================================================
    // UI STATE
    // ================================================================

    [ObservableProperty]
    private bool isBusy;

    [ObservableProperty]
    private string statusMessage = "Ready";

    [ObservableProperty]
    private ApiConnectionState connectionState = ApiConnectionState.Stopped;

    [ObservableProperty]
    private string connectionStatusMessage = "API is not connected.";

    [ObservableProperty]
    private DateTimeOffset? lastHeartbeatUtc;

    public bool IsApiConnected =>
        ConnectionState == ApiConnectionState.Connected;

    public string ApiConnectionStatusText => ConnectionState switch
    {
        ApiConnectionState.Connected => "API connected",
        ApiConnectionState.Connecting => "Connecting to API",
        ApiConnectionState.Reconnecting => "Reconnecting to API",
        ApiConnectionState.Disconnected => "API disconnected",
        _ => "API stopped"
    };

    private string _themeMode = "System";

    public string ThemeMode
    {
        get => _themeMode;
        set
        {
            if (!SetProperty(ref _themeMode, value))
                return;

            OnPropertyChanged(nameof(ThemeModeDisplay));
        }
    }

    public string ThemeModeDisplay => ThemeMode switch
    {
        "Light" => "Light",
        "Dark" => "Dark",
        _ => "System"
    };

    [RelayCommand]
    private void CycleTheme()
    {
        ThemeMode = ThemeMode switch
        {
            "System" => "Light",
            "Light" => "Dark",
            _ => "System"
        };
    }

    [ObservableProperty]
    private BitmapImage? hoveredMaskImage;

    [ObservableProperty]
    private double autoEnhanceStrength = 1.0;

    [ObservableProperty]
    private double manualExposure;

    [ObservableProperty]
    private double manualContrast;

    [ObservableProperty]
    private double manualHighlights;

    [ObservableProperty]
    private double manualShadows;

    [ObservableProperty]
    private double manualTemperature;

    [ObservableProperty]
    private double manualTint;

    [ObservableProperty]
    private double manualSaturation;

    [ObservableProperty]
    private double manualSharpness;

    private bool _synchronizingManualControls;

    private bool _suppressLivePreview;

    partial void OnManualExposureChanged(double value)
    {
        if (_synchronizingManualControls)
            return;

        double clamped = ClampManualValue(value, -2.0, 2.0);
        if (Math.Abs(clamped - value) > 0.0001)
            ManualExposure = clamped;

        _ = TriggerLivePreviewAsync(isManual: true);
    }

    partial void OnManualContrastChanged(double value)
    {
        if (_synchronizingManualControls)
            return;

        double clamped = ClampManualValue(value, -100.0, 100.0);
        if (Math.Abs(clamped - value) > 0.0001)
            ManualContrast = clamped;

        _ = TriggerLivePreviewAsync(isManual: true);
    }

    partial void OnManualHighlightsChanged(double value)
    {
        if (_synchronizingManualControls)
            return;

        double clamped = ClampManualValue(value, -100.0, 100.0);
        if (Math.Abs(clamped - value) > 0.0001)
            ManualHighlights = clamped;

        _ = TriggerLivePreviewAsync(isManual: true);
    }

    partial void OnManualShadowsChanged(double value)
    {
        if (_synchronizingManualControls)
            return;

        double clamped = ClampManualValue(value, -100.0, 100.0);
        if (Math.Abs(clamped - value) > 0.0001)
            ManualShadows = clamped;

        _ = TriggerLivePreviewAsync(isManual: true);
    }

    partial void OnManualTemperatureChanged(double value)
    {
        if (_synchronizingManualControls)
            return;

        double clamped = ClampManualValue(value, -100.0, 100.0);
        if (Math.Abs(clamped - value) > 0.0001)
            ManualTemperature = clamped;

        _ = TriggerLivePreviewAsync(isManual: true);
    }

    partial void OnManualTintChanged(double value)
    {
        if (_synchronizingManualControls)
            return;

        double clamped = ClampManualValue(value, -100.0, 100.0);
        if (Math.Abs(clamped - value) > 0.0001)
            ManualTint = clamped;

        _ = TriggerLivePreviewAsync(isManual: true);
    }

    partial void OnManualSaturationChanged(double value)
    {
        if (_synchronizingManualControls)
            return;

        double clamped = ClampManualValue(value, -100.0, 100.0);
        if (Math.Abs(clamped - value) > 0.0001)
            ManualSaturation = clamped;

        _ = TriggerLivePreviewAsync(isManual: true);
    }

    partial void OnManualSharpnessChanged(double value)
    {
        if (_synchronizingManualControls)
            return;

        double clamped = ClampManualValue(value, 0.0, 100.0);
        if (Math.Abs(clamped - value) > 0.0001)
            ManualSharpness = clamped;

        _ = TriggerLivePreviewAsync(isManual: true);
    }

    partial void OnAutoEnhanceStrengthChanged(double value)
    {
        if (_suppressLivePreview)
            return;

        double clamped = ClampManualValue(value, 0.0, 1.0);
        if (Math.Abs(clamped - value) > 0.0001)
            AutoEnhanceStrength = clamped;

        _ = TriggerLivePreviewAsync(isManual: false);
    }

    private CancellationTokenSource? _livePreviewCts;

    private async Task TriggerLivePreviewAsync(bool isManual)
    {
        if (IsBusy)
            return;

        string? sourcePath = GetPreviewSourcePath();
        if (string.IsNullOrWhiteSpace(sourcePath) || !File.Exists(sourcePath))
            return;

        _livePreviewCts?.Cancel();
        _livePreviewCts = new CancellationTokenSource();
        CancellationToken token = _livePreviewCts.Token;

        try
        {
            await Task.Delay(180, token);
            if (token.IsCancellationRequested)
                return;

            if (isManual)
                await ApplyManualPreviewAsync(token);
            else
                await ApplyAutoEnhancePreviewAsync(token);
        }
        catch (OperationCanceledException)
        {
        }
        catch (Exception)
        {
        }
    }

    private async Task ApplyManualPreviewAsync(CancellationToken token)
    {
        string? sourcePath = GetPreviewSourcePath();
        if (string.IsNullOrWhiteSpace(sourcePath) || !File.Exists(sourcePath))
            return;

        string normalizedSourcePath = Path.GetFullPath(sourcePath);

        IReadOnlyCollection<string>? selectedMasks =
            ManualAdjustMask
                ? ParseSelectedMaskNames()
                : null;

        if (ManualAdjustMask && (selectedMasks == null || selectedMasks.Count == 0))
            return;

        JsonElement maskJson = default;
        if (ManualAdjustMask)
        {
            string? cachedMaskJson = GetReusableMaskJson();
            if (string.IsNullOrWhiteSpace(cachedMaskJson))
                return;

            using JsonDocument maskDocument = JsonDocument.Parse(cachedMaskJson);
            maskJson = maskDocument.RootElement.Clone();
        }

        var settings = new ManualAdjustmentSettings
        {
            Exposure = ManualExposure,
            Contrast = ManualContrast,
            Highlights = ManualHighlights,
            Shadows = ManualShadows,
            Temperature = ManualTemperature,
            Tint = ManualTint,
            Saturation = ManualSaturation,
            Sharpness = ManualSharpness,
            UseMask = ManualAdjustMask,
            MaskJson = maskJson,
            MaskNames = selectedMasks?.ToArray() ?? Array.Empty<string>(),
            Feather = MaskedFeather
        };

        string apiInputPath = await PrepareApiInputAsync(normalizedSourcePath, token);
        byte[] imageBytes = await File.ReadAllBytesAsync(apiInputPath, token);

        AutoEnhanceApiResult result = await _api.ManualAdjustAsync(
            imageBytes,
            settings,
            jobId: Guid.NewGuid().ToString("N"),
            cancellationToken: token);

        token.ThrowIfCancellationRequested();

        string outputPath = Path.Combine(
            Path.GetTempPath(),
            "AutoPhotoEditor",
            "LivePreview",
            $"{Guid.NewGuid():N}_manual.png");

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        await File.WriteAllBytesAsync(outputPath, result.ImageBytes, token);
        SetLivePreviewImage(outputPath);
        ComparisonPosition = 50;
    }

    private async Task ApplyAutoEnhancePreviewAsync(CancellationToken token)
    {
        string? sourcePath = GetPreviewSourcePath();
        if (string.IsNullOrWhiteSpace(sourcePath) || !File.Exists(sourcePath))
            return;

        string apiInputPath = await PrepareApiInputAsync(sourcePath, token);
        byte[] imageBytes = await File.ReadAllBytesAsync(apiInputPath, token);

        string jobId = Guid.NewGuid().ToString("N");
        AutoEnhanceDataResult analysisResult = await _api.ExtractImageDataAsync(
            imageBytes,
            jobId,
            cancellationToken: token,
            summaryOnly: false);

        token.ThrowIfCancellationRequested();

        AutoEnhanceApiResult result = await _api.AutoEnhanceAsync(
            imageBytes,
            analysisResult.Analysis,
            strength: AutoEnhanceStrength,
            jobId: Guid.NewGuid().ToString("N"),
            cancellationToken: token);

        token.ThrowIfCancellationRequested();

        string outputPath = Path.Combine(
            Path.GetTempPath(),
            "AutoPhotoEditor",
            "LivePreview",
            $"{Guid.NewGuid():N}_auto.png");

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        await File.WriteAllBytesAsync(outputPath, result.ImageBytes, token);
        SetLivePreviewImage(outputPath);
        ComparisonPosition = 50;
    }

    partial void OnMaskedStrengthChanged(double value)
    {
        double clamped = ClampManualValue(value, 0.0, 1.0);
        if (Math.Abs(clamped - value) > 0.0001)
            MaskedStrength = clamped;
    }

    partial void OnMaskedFeatherChanged(double value)
    {
        double clamped = ClampManualValue(value, 0.0, 100.0);
        if (Math.Abs(clamped - value) > 0.0001)
            MaskedFeather = clamped;
    }

    partial void OnGeometryScaleChanged(double value)
    {
        double clamped = ClampManualValue(value, 0.01, 1000.0);
        if (Math.Abs(clamped - value) > 0.0001)
            GeometryScale = clamped;
    }

    partial void OnLensStrengthChanged(double value)
    {
        double clamped = ClampManualValue(value, 0.0, 1.0);
        if (Math.Abs(clamped - value) > 0.0001)
            LensStrength = clamped;
    }

    partial void OnLensCenterXChanged(double value)
    {
        double clamped = ClampManualValue(value, 0.0, 1.0);
        if (Math.Abs(clamped - value) > 0.0001)
            LensCenterX = clamped;
    }

    partial void OnLensCenterYChanged(double value)
    {
        double clamped = ClampManualValue(value, 0.0, 1.0);
        if (Math.Abs(clamped - value) > 0.0001)
            LensCenterY = clamped;
    }

    partial void OnLensFocalScaleChanged(double value)
    {
        double clamped = ClampManualValue(value, 0.01, 100.0);
        if (Math.Abs(clamped - value) > 0.0001)
            LensFocalScale = clamped;
    }

    private static double ClampManualValue(
        double value,
        double minimum,
        double maximum)
    {
        return
            double.IsFinite(value)
                ? Math.Clamp(value, minimum, maximum)
                : minimum <= 0.0 && maximum >= 0.0
                    ? 0.0
                    : minimum;
    }

    private static double ClampAutoEnhanceStrength(double value)
    {
        return Math.Clamp(double.IsFinite(value) ? Math.Abs(value) : 0.0, 0.0, 1.0);
    }

    private static (double LuminosityRecommendedEv, double NeutralConfidence)? TryGetBestEnhancement(JsonElement analysis)
    {
        if (analysis.ValueKind != JsonValueKind.Object)
            return null;

        if (!analysis.TryGetProperty("best_enhancement", out JsonElement bestEnhancement) ||
            bestEnhancement.ValueKind != JsonValueKind.Object)
        {
            return null;
        }

        if (!bestEnhancement.TryGetProperty("luminosity", out JsonElement luminosity) ||
            luminosity.ValueKind != JsonValueKind.Object)
        {
            return null;
        }

        double luminosityChange = 0.0;
        if (luminosity.TryGetProperty("recommended_change_ev", out JsonElement ev) && ev.ValueKind == JsonValueKind.Number)
            luminosityChange = ev.GetDouble();

        double neutralConfidence = 0.0;
        if (bestEnhancement.TryGetProperty("neutral_tone", out JsonElement neutralTone) &&
            neutralTone.ValueKind == JsonValueKind.Object &&
            neutralTone.TryGetProperty("confidence", out JsonElement confidence) &&
            confidence.ValueKind == JsonValueKind.Number)
        {
            neutralConfidence = confidence.GetDouble();
        }

        return (luminosityChange, neutralConfidence);
    }

    [ObservableProperty]
    private string maskDevice = "auto";

    [ObservableProperty]
    private double maskedStrength = 1.0;

    [ObservableProperty]
    private double maskedFeather = 2.0;

    [ObservableProperty]
    private string selectedMaskNamesText = string.Empty;

    private string? _selectedMaskName;

    public string? SelectedMaskName
    {
        get => _selectedMaskName;
        set
        {
            if (SetProperty(ref _selectedMaskName, value) &&
                !string.IsNullOrWhiteSpace(value))
            {
                SelectedMaskNamesText = value;
                ApplySelectedMaskRecommendation(value);
            }
        }
    }

    private bool _manualAdjustMask;

    public bool ManualAdjustMask
    {
        get => _manualAdjustMask;
        set => SetProperty(ref _manualAdjustMask, value);
    }

    public bool HasAvailableMasks => AvailableMaskNames.Count > 0;

    public string MaskSelectionHint => HasAvailableMasks
        ? $"{AvailableMaskNames.Count} mask region(s) ready. Select one or more to enhance."
        : "Mask regions are loaded automatically after an image is opened.";

    [ObservableProperty]
    private string denoiseDevice = "cpu";

    [ObservableProperty]
    private string geometryMode = "auto";

    [ObservableProperty]
    private double geometryRotate;

    [ObservableProperty]
    private double geometryAspect;

    [ObservableProperty]
    private double geometryScale = 100.0;

    [ObservableProperty]
    private double geometryX;

    [ObservableProperty]
    private double geometryY;

    [ObservableProperty]
    private bool geometryCrop = true;

    [ObservableProperty]
    private bool lensManual;

    [ObservableProperty]
    private double lensK1;

    [ObservableProperty]
    private double lensK2;

    [ObservableProperty]
    private double lensK3;

    [ObservableProperty]
    private double lensP1;

    [ObservableProperty]
    private double lensP2;

    [ObservableProperty]
    private double lensStrength = 1.0;

    [ObservableProperty]
    private double lensCenterX = 0.5;

    [ObservableProperty]
    private double lensCenterY = 0.5;

    [ObservableProperty]
    private double lensFocalScale = 1.0;

    [ObservableProperty]
    private bool lensNoCrop;

    [ObservableProperty]
    private string lensCameraMaker = string.Empty;

    [ObservableProperty]
    private string lensCameraModel = string.Empty;

    [ObservableProperty]
    private string lensMaker = string.Empty;

    [ObservableProperty]
    private string lensModel = string.Empty;

    [ObservableProperty]
    private string pipelineStages = "analyze,auto-enhance";

    public ObservableCollection<string> AvailableMaskNames { get; } = new();

    public ObservableCollection<MaskOption> MaskOptions { get; } = new();

    public void SetHoveredMask(MaskOption? option)
    {
        if (option == null || string.IsNullOrWhiteSpace(option.BinaryPngBase64))
        {
            HoveredMaskImage = null;
            return;
        }

        try
        {
            byte[] bytes = Convert.FromBase64String(option.BinaryPngBase64);
            using var stream = new MemoryStream(bytes);
            var bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.StreamSource = stream;
            bitmap.EndInit();
            bitmap.Freeze();
            HoveredMaskImage = bitmap;
        }
        catch (FormatException)
        {
            HoveredMaskImage = null;
        }
        catch (ArgumentException)
        {
            HoveredMaskImage = null;
        }
        catch (InvalidOperationException)
        {
            HoveredMaskImage = null;
        }
    }

    private void ApplySelectedMaskRecommendation(string maskName)
    {
        string? maskJson = GetReusableMaskJson();
        if (string.IsNullOrWhiteSpace(maskJson))
            return;

        try
        {
            using JsonDocument document = JsonDocument.Parse(maskJson);
            if (!document.RootElement.TryGetProperty("masks", out JsonElement masks) ||
                masks.ValueKind != JsonValueKind.Object ||
                !masks.TryGetProperty(maskName, out JsonElement mask) ||
                !mask.TryGetProperty("enhancement", out JsonElement enhancement) ||
                enhancement.ValueKind != JsonValueKind.Object ||
                !enhancement.TryGetProperty("adjustments", out JsonElement adjustments))
            {
                return;
            }

            _synchronizingManualControls = true;
            ManualExposure = GetNumber(adjustments, "exposure", 0.0);
            ManualContrast = GetNumber(adjustments, "contrast", 0.0);
            ManualHighlights = GetNumber(adjustments, "highlights", 0.0);
            ManualShadows = GetNumber(adjustments, "shadows", 0.0);
            ManualTemperature = GetNumber(adjustments, "temperature", 0.0);
            ManualTint = GetNumber(adjustments, "tint", 0.0);
            ManualSaturation = GetNumber(adjustments, "saturation", 0.0);
            ManualSharpness = GetNumber(adjustments, "clarity", 0.0);
        }
        catch (JsonException)
        {
        }
        finally
        {
            _synchronizingManualControls = false;
        }
    }


    // ================================================================
    // COMPARISON
    // ================================================================

    [ObservableProperty]
    private double comparisonPosition = 50.0;


    partial void OnComparisonPositionChanged(double value)
    {
        double clamped = Math.Clamp(value, 0.0, 100.0);

        if (Math.Abs(clamped - value) > 0.001)
        {
            ComparisonPosition = clamped;
        }
    }


    // ================================================================
    // HISTORY
    // ================================================================

    public ObservableCollection<PhotoEditJob> JobHistory { get; }
        = new();

    [ObservableProperty]
    private int historyIndex = -1;

    public bool CanGoPrevious => HistoryIndex >= 0;

    public bool CanGoNext => HistoryIndex < JobHistory.Count - 1;

    public string HistoryPositionText =>
        HistoryIndex < 0
            ? $"Original  •  0 / {JobHistory.Count}"
            : $"Edit {HistoryIndex + 1} / {JobHistory.Count}";

    partial void OnHistoryIndexChanged(int value)
    {
        OnPropertyChanged(nameof(CanGoPrevious));
        OnPropertyChanged(nameof(CanGoNext));
        OnPropertyChanged(nameof(HistoryPositionText));

        ApplyHistoryControls(value);
    }


    // ================================================================
    // CONSTRUCTOR
    // ================================================================

    public MainViewModel(
        AutoPhotoEditorApiClient api,
        ApiWebSocketConnection apiConnection)
    {
        _api = api;
        _apiConnection = apiConnection;
        _apiConnection.StatusChanged += ApiConnection_StatusChanged;

        LUTs.Add(
            new LUTItem
            {
                Name = "Cinematic Teal",
                FilePath = "luts/teal_orange.cube"
            });

        LUTs.Add(
            new LUTItem
            {
                Name = "Vintage Film",
                FilePath = "luts/vintage.cube"
            });

        SelectedLut = LUTs.FirstOrDefault();
    }

    public Task StartApiConnectionAsync(
        CancellationToken cancellationToken = default)
    {
        return _apiConnection.StartAsync(cancellationToken);
    }

    public async Task StopApiConnectionAsync()
    {
        await _apiConnection.StopAsync();
    }

    private void ApiConnection_StatusChanged(
        object? sender,
        ApiConnectionStatusChangedEventArgs e)
    {
        if (Application.Current?.Dispatcher is { } dispatcher &&
            !dispatcher.CheckAccess())
        {
            dispatcher.BeginInvoke(
                new Action(
                    () => ApplyApiConnectionStatus(e)));
            return;
        }

        ApplyApiConnectionStatus(e);
    }

    private void ApplyApiConnectionStatus(
        ApiConnectionStatusChangedEventArgs e)
    {
        ConnectionState = e.State;
        ConnectionStatusMessage = e.Message;
        LastHeartbeatUtc = e.LastHeartbeatUtc;
        NotifyConnectionDependentPropertiesChanged();
    }

    private void NotifyConnectionDependentPropertiesChanged()
    {
        OnPropertyChanged(nameof(IsApiConnected));
        OnPropertyChanged(nameof(ApiConnectionStatusText));
    }


    // ================================================================
    // SHOW BEFORE
    // ================================================================

    [RelayCommand]
    private void ShowBefore()
    {
        if (OriginalImage == null)
        {
            StatusMessage = "Choose an image first.";
            return;
        }

        ComparisonPosition = 0;
    }


    // ================================================================
    // SHOW AFTER
    // ================================================================

    [RelayCommand]
    private void ShowAfter()
    {
        if (EditedImage == null)
        {
            StatusMessage = "Apply an edit before showing the after image.";
            return;
        }

        ComparisonPosition = 100;
    }


    // ================================================================
    // RESET COMPARISON
    // ================================================================

    [RelayCommand]
    private void ResetComparison()
    {
        ComparisonPosition = 50;
    }


    // ================================================================
    // CHOOSE IMAGE
    // ================================================================

    [RelayCommand]
    private void ChooseImage()
    {
        var dialog = new OpenFileDialog
        {
            Title = "Choose an image",
            Filter =
                "Image Files|*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.tif;*.tiff"
        };

        if (dialog.ShowDialog() != true)
            return;

        try
        {
            string path = dialog.FileName;

            // --------------------------------------------------------
            // ALWAYS LOAD THE ORIGINAL FROM THE USER'S FILE.
            // NEVER USE THE API PREVIEW HERE.
            // --------------------------------------------------------

            BitmapImage original =
                LoadDisplayPreviewBitmap(
                    path,
                    DisplayPreviewMaxDimension);

            OriginalImage = original;

            OriginalImagePath = path;


            // --------------------------------------------------------
            // RESET EDITED STATE
            // --------------------------------------------------------

            EditedImage = null;
            EditedImagePath = null;


            // --------------------------------------------------------
            // RESET HISTORY
            // --------------------------------------------------------

            JobHistory.Clear();

            HistoryIndex = -1;

            OnPropertyChanged(nameof(CanGoPrevious));
            OnPropertyChanged(nameof(CanGoNext));
            OnPropertyChanged(nameof(HistoryPositionText));


            // --------------------------------------------------------
            // RESET MASK DATA
            // --------------------------------------------------------

            _maskJsonForOriginal = null;

            _analysisJsonForOriginal = null;

            _maskImagePath = null;

            _analysisImagePath = null;

            AvailableMaskNames.Clear();

            SelectedMaskNamesText = string.Empty;
            SelectedMaskName = null;
            ManualAdjustMask = false;
            MaskOptions.Clear();
            HoveredMaskImage = null;


            // --------------------------------------------------------
            // RESET COMPARISON
            // --------------------------------------------------------

            ComparisonPosition = 50;


            StatusMessage = "Image loaded.";
            _ = PreloadImageDataAsync(path);
        }
        catch (Exception ex)
        {
            StatusMessage =
                $"Unable to load image: {ex.Message}";
        }
    }

    private bool IsCurrentOriginalImage(string imagePath)
    {
        return string.Equals(
            OriginalImagePath,
            imagePath,
            StringComparison.OrdinalIgnoreCase);
    }

    private void ApplyRecommendedAutoEnhanceStrength(JsonElement analysis)
    {
        var bestEnhancement = TryGetBestEnhancement(analysis);
        if (bestEnhancement != null)
        {
            _suppressLivePreview = true;
            try
            {
                AutoEnhanceStrength = ClampAutoEnhanceStrength(bestEnhancement.Value.LuminosityRecommendedEv);
            }
            finally
            {
                _suppressLivePreview = false;
            }
        }
    }


    // ================================================================
    // AUTO ENHANCE
    // ================================================================

    private async Task PreloadImageDataAsync(string imagePath)
    {
        _imageDataPreloadCts?.Cancel();
        _imageDataPreloadCts?.Dispose();
        _imageDataPreloadCts = new CancellationTokenSource();
        CancellationToken token = _imageDataPreloadCts.Token;

        bool analysisReady = false;
        bool masksReady = false;

        try
        {
            string apiInputPath = await PrepareApiInputAsync(imagePath, token);
            byte[] analysisBytes = await File.ReadAllBytesAsync(apiInputPath, token);

            try
            {
                AutoEnhanceDataResult analysisResult =
                    await _api.ExtractImageDataAsync(
                        analysisBytes,
                        Guid.NewGuid().ToString("N"),
                        cancellationToken: token,
                        summaryOnly: false);

                token.ThrowIfCancellationRequested();

                if (!IsCurrentOriginalImage(imagePath))
                    return;

                _analysisJsonForOriginal = analysisResult.AnalysisJson;
                _analysisImagePath = imagePath;
                analysisReady = true;
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch
            {
            }

            token.ThrowIfCancellationRequested();

            if (!IsCurrentOriginalImage(imagePath))
                return;

            byte[] maskBytes = await File.ReadAllBytesAsync(imagePath, token);

            try
            {
                MaskDataResult maskResult =
                    await _api.MaskDataAsync(
                        maskBytes,
                        Guid.NewGuid().ToString("N"),
                        cancellationToken: token,
                        device: MaskDevice);

                token.ThrowIfCancellationRequested();

                if (!IsCurrentOriginalImage(imagePath))
                    return;

                if (!string.IsNullOrWhiteSpace(maskResult.MaskJsonText))
                {
                    _maskJsonForOriginal = maskResult.MaskJsonText;
                    _maskImagePath = imagePath;
                    UpdateMaskOptions(
                        maskResult.MaskNames,
                        maskResult.MaskBinaryPngBase64);
                    masksReady = true;
                }
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch
            {
            }

            if (!IsCurrentOriginalImage(imagePath))
                return;

            StatusMessage = analysisReady && masksReady
                ? "Image loaded. AI and mask data ready."
                : analysisReady
                    ? "Image loaded. AI data ready; mask data will be fetched when needed."
                    : masksReady
                        ? "Image loaded. Mask data ready; AI data will be fetched when needed."
                        : "Image loaded. AI and mask data will be fetched when needed.";
        }
        catch (OperationCanceledException)
        {
        }
        catch
        {
            if (IsCurrentOriginalImage(imagePath))
                StatusMessage = analysisReady || masksReady
                    ? "Image loaded. Some AI data will be fetched when needed."
                    : "Image loaded. AI and mask data will be fetched when needed.";
        }
    }

    [RelayCommand]
    private async Task AutoEnhanceAsync()
    {
        string? sourcePath = GetPreviewSourcePath();

        if (string.IsNullOrWhiteSpace(sourcePath))
        {
            StatusMessage = "Choose an image first.";
            return;
        }

        if (!File.Exists(sourcePath))
        {
            StatusMessage =
                "The selected image no longer exists.";
            return;
        }

        StartProcessing("Analysing image...");

        try
        {
            _processingCts =
                new CancellationTokenSource();

            CancellationToken token =
                _processingCts.Token;


            // --------------------------------------------------------
            // CREATE TEMPORARY API PREVIEW ONLY
            // --------------------------------------------------------

            string apiInputPath =
                await PrepareApiInputAsync(
                    sourcePath,
                    token);

            token.ThrowIfCancellationRequested();


            byte[] imageBytes =
                await File.ReadAllBytesAsync(
                    apiInputPath,
                    token);


            string jobId =
                Guid.NewGuid().ToString("N");

            _activeJobId =
                jobId;


            // --------------------------------------------------------
            // ANALYSIS
            // --------------------------------------------------------

            StatusMessage =
                "Analysing image...";

            JsonElement analysis;
            if (IsCurrentOriginalImage(sourcePath) &&
                string.Equals(_analysisImagePath, sourcePath, StringComparison.OrdinalIgnoreCase) &&
                !string.IsNullOrWhiteSpace(_analysisJsonForOriginal))
            {
                using JsonDocument cachedAnalysis = JsonDocument.Parse(_analysisJsonForOriginal);
                analysis = cachedAnalysis.RootElement.Clone();
                ApplyRecommendedAutoEnhanceStrength(analysis);
            }
            else
            {
                var summaryResult = await _api.ExtractImageDataAsync(
                    imageBytes,
                    Guid.NewGuid().ToString("N"),
                    OnApiProgress,
                    token,
                    summaryOnly: true);

                token.ThrowIfCancellationRequested();

                ApplyRecommendedAutoEnhanceStrength(summaryResult.Analysis);

                var analysisResult = await _api.ExtractImageDataAsync(
                    imageBytes,
                    jobId,
                    OnApiProgress,
                    token);
                analysis = analysisResult.Analysis;
            }

            // --------------------------------------------------------
            // ENHANCE
            // --------------------------------------------------------

            StatusMessage =
                "Enhancing image...";

            var result =
                await _api.AutoEnhanceAsync(
                    imageBytes,
                    analysis,
                    strength: AutoEnhanceStrength,
                    jobId,
                    OnApiProgress,
                    token);

            token.ThrowIfCancellationRequested();


            // --------------------------------------------------------
            // SAVE EDITED RESULT
            // --------------------------------------------------------

            string outputDirectory =
                Path.Combine(
                    Path.GetTempPath(),
                    "AutoPhotoEditor",
                    "Generated");

            Directory.CreateDirectory(outputDirectory);


            string outputFile =
                Path.Combine(
                    outputDirectory,
                    $"{jobId}_auto.png");


            await File.WriteAllBytesAsync(
                outputFile,
                result.ImageBytes,
                token);


            // --------------------------------------------------------
            // IMPORTANT:
            // ONLY EDITED IMAGE IS UPDATED.
            //
            // OriginalImage remains untouched.
            // --------------------------------------------------------

            SetEditedImage(outputFile);
            string analysisJson = analysis.GetRawText();
            ApplyAutoAnalysisSettings(analysisJson);


            AddHistory(
                outputFile,
                "Auto Enhance",
                settingsJson: JsonSerializer.Serialize(new
                {
                    strength = AutoEnhanceStrength
                }),
                analysisJson: analysisJson);


            ComparisonPosition = 50;

            StatusMessage =
                "Auto enhancement complete.";
        }
        catch (OperationCanceledException)
        {
            StatusMessage =
                "Processing cancelled.";
        }
        catch (Exception ex)
        {
            StatusMessage =
                $"Auto Enhance failed: {ex.Message}";
        }
        finally
        {
            StopProcessing();
        }
    }


    // ================================================================
    // MASKED AUTO ENHANCE
    // ================================================================

    [RelayCommand]
    private async Task MaskedEnhanceAsync()
    {
        // ------------------------------------------------------------
        // Preview processing uses the current preview image. The backend
        // resizes the reusable mask data to the input geometry; export replay
        // later sends the untouched original at full resolution.
        // ------------------------------------------------------------

        string? sourcePath =
            GetPreviewSourcePath();
        string? originalPath =
            OriginalImagePath;


        if (string.IsNullOrWhiteSpace(sourcePath) ||
            !File.Exists(sourcePath) ||
            string.IsNullOrWhiteSpace(originalPath) ||
            !File.Exists(originalPath))
        {
            StatusMessage =
                "Choose an image first.";

            return;
        }


        StartProcessing(
            "Preparing masked enhancement...");


        try
        {
            _processingCts =
                new CancellationTokenSource();

            CancellationToken token =
                _processingCts.Token;


            string apiInputPath =
                await PrepareApiInputAsync(
                    sourcePath,
                    token);

            byte[] imageBytes =
                await File.ReadAllBytesAsync(
                    apiInputPath,
                    token);


            string jobId =
                Guid.NewGuid().ToString("N");

            _activeJobId =
                jobId;


            // --------------------------------------------------------
            // MASK DATA
            // --------------------------------------------------------

            string? maskJson = GetReusableMaskJson();

            if (!string.IsNullOrWhiteSpace(maskJson))
            {
                _maskJsonForOriginal =
                    maskJson;
                _maskImagePath = originalPath;
            }


            if (string.IsNullOrWhiteSpace(maskJson))
            {
                StatusMessage =
                    "Analysing image masks...";

                byte[] maskImageBytes =
                    await File.ReadAllBytesAsync(
                        originalPath,
                        token);


                MaskDataResult maskResult =
                    await _api.MaskDataAsync(
                        maskImageBytes,
                        jobId,
                        OnApiProgress,
                        token,
                        device: MaskDevice);


                maskJson =
                    maskResult.MaskJsonText;


                if (string.IsNullOrWhiteSpace(maskJson))
                {
                    throw new AutoPhotoEditorApiException(
                        "The API returned no reusable mask data.");
                }


                // Only cache masks when they belong to
                // the ORIGINAL image.

                _maskJsonForOriginal =
                    maskJson;
                _maskImagePath = originalPath;

                UpdateMaskOptions(
                    maskResult.MaskNames,
                    maskResult.MaskBinaryPngBase64);
            }


            using JsonDocument maskDocument =
                JsonDocument.Parse(maskJson);

            if (AvailableMaskNames.Count == 0)
            {
                UpdateMaskOptions(maskDocument.RootElement);
            }

            IReadOnlyCollection<string>? selectedMaskNames = ParseSelectedMaskNames();
            if (selectedMaskNames == null || selectedMaskNames.Count == 0)
            {
                StatusMessage = "Choose at least one mask before enhancing.";
                return;
            }


            // --------------------------------------------------------
            // ENHANCE
            // --------------------------------------------------------

            StatusMessage =
                "Enhancing masked regions...";


            AutoEnhanceApiResult result =
                await _api.MaskedEnhanceAsync(
                    imageBytes,
                    maskDocument.RootElement,
                    strength: MaskedStrength,
                    feather: MaskedFeather,
                    maskNames: selectedMaskNames,
                    jobId: jobId,
                    progress: OnApiProgress,
                    cancellationToken: token);


            token.ThrowIfCancellationRequested();


            // --------------------------------------------------------
            // SAVE EDITED RESULT
            // --------------------------------------------------------

            string outputDirectory =
                Path.Combine(
                    Path.GetTempPath(),
                    "AutoPhotoEditor",
                    "Generated");


            Directory.CreateDirectory(
                outputDirectory);


            string outputPath =
                Path.Combine(
                    outputDirectory,
                    $"{jobId}_masked.png");


            await File.WriteAllBytesAsync(
                outputPath,
                result.ImageBytes,
                token);


            // --------------------------------------------------------
            // ONLY UPDATE EDITED IMAGE
            // --------------------------------------------------------

            SetEditedImage(outputPath);


            AddHistory(
                outputPath,
                "Masked Enhance",
                maskJson,
                settingsJson: JsonSerializer.Serialize(new
                {
                    strength = MaskedStrength,
                    feather = MaskedFeather,
                    mask_names = selectedMaskNames,
                    mask_report = result.MaskReport.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null
                        ? null
                        : (object)result.MaskReport
                }),
                isMaskScoped: true,
                maskNames: selectedMaskNames);


            ComparisonPosition = 50;


            StatusMessage =
                "Original photo masked enhancement complete.";
        }
        catch (OperationCanceledException)
        {
            StatusMessage =
                "Processing cancelled.";
        }
        catch (Exception ex)
        {
            StatusMessage =
                $"Masked enhancement failed: {ex.Message}";
        }
        finally
        {
            StopProcessing();
        }
    }


    // ================================================================
    // MANUAL ADJUSTMENT
    // ================================================================

    [RelayCommand]
    private Task ManualAdjustAsync()
    {
        IReadOnlyCollection<string>? selectedMasks =
            ManualAdjustMask
                ? ParseSelectedMaskNames()
                : null;

        if (ManualAdjustMask && (selectedMasks == null || selectedMasks.Count == 0))
        {
            StatusMessage = "Choose at least one mask before applying a masked adjustment.";
            return Task.CompletedTask;
        }

        JsonElement maskJson = default;
        if (ManualAdjustMask)
        {
            string? cachedMaskJson = GetReusableMaskJson();
            if (string.IsNullOrWhiteSpace(cachedMaskJson))
            {
                StatusMessage = "Extract mask data before applying a masked adjustment.";
                return Task.CompletedTask;
            }

            try
            {
                using JsonDocument maskDocument = JsonDocument.Parse(cachedMaskJson);
                maskJson = maskDocument.RootElement.Clone();
            }
            catch (JsonException)
            {
                StatusMessage = "The cached mask data is invalid. Extract masks again.";
                return Task.CompletedTask;
            }
        }

        var settings = new ManualAdjustmentSettings
        {
            Exposure = ManualExposure,
            Contrast = ManualContrast,
            Highlights = ManualHighlights,
            Shadows = ManualShadows,
            Temperature = ManualTemperature,
            Tint = ManualTint,
            Saturation = ManualSaturation,
            Sharpness = ManualSharpness,
            UseMask = ManualAdjustMask,
            MaskJson = maskJson,
            MaskNames = selectedMasks?.ToArray() ?? Array.Empty<string>(),
            Feather = MaskedFeather
        };

        string settingsJson = JsonSerializer.Serialize(new
        {
            target = ManualAdjustMask ? "mask" : "image",
            mask_names = selectedMasks?.ToArray() ?? Array.Empty<string>(),
            feather = MaskedFeather,
            adjustments = new
            {
                exposure = ManualExposure,
                contrast = ManualContrast,
                highlights = ManualHighlights,
                shadows = ManualShadows,
                temperature = ManualTemperature,
                tint = ManualTint,
                saturation = ManualSaturation,
                sharpness = ManualSharpness
            }
        });

        return RunImageOperationAsync(
            "Applying manual adjustments...",
            "Manual Adjust",
            "manual",
            (imageBytes, jobId, token) =>
                _api.ManualAdjustAsync(
                    imageBytes,
                    settings,
                    jobId: jobId,
                    progress: OnApiProgress,
                    cancellationToken: token),
            settingsJson: settingsJson,
            maskJson: ManualAdjustMask ? maskJson.GetRawText() : null,
            maskNames: selectedMasks,
            isMaskScoped: ManualAdjustMask);
    }


    // ================================================================
    // COLOR GRADE
    // ================================================================

    [RelayCommand]
    private async Task ColorGradeAsync()
    {
        string? sourcePath =
            GetPreviewSourcePath();


        if (string.IsNullOrWhiteSpace(sourcePath))
        {
            StatusMessage =
                "Choose an image first.";

            return;
        }


        if (SelectedLut == null)
        {
            StatusMessage =
                "Choose a color grade first.";

            return;
        }


        if (!File.Exists(sourcePath))
        {
            StatusMessage =
                "The selected image no longer exists.";

            return;
        }


        StartProcessing(
            "Applying color grade...");


        try
        {
            _processingCts =
                new CancellationTokenSource();

            CancellationToken token =
                _processingCts.Token;


            string apiInputPath =
                await PrepareApiInputAsync(
                    sourcePath,
                    token);


            token.ThrowIfCancellationRequested();


            byte[] imageBytes =
                await File.ReadAllBytesAsync(
                    apiInputPath,
                    token);


            string jobId =
                Guid.NewGuid().ToString("N");

            _activeJobId =
                jobId;


            StatusMessage =
                "Applying color grade...";


            var result =
                await _api.ApplyPresetAsync(
                    imageBytes,
                    SelectedLut.FilePath,
                    jpegQuality: 97,
                    jobId: jobId,
                    progress: OnApiProgress,
                    cancellationToken: token);


            token.ThrowIfCancellationRequested();


            string outputDirectory =
                Path.Combine(
                    Path.GetTempPath(),
                    "AutoPhotoEditor",
                    "Generated");


            Directory.CreateDirectory(
                outputDirectory);


            string outputPath =
                Path.Combine(
                    outputDirectory,
                    $"{jobId}_lut.png");


            await File.WriteAllBytesAsync(
                outputPath,
                result.ImageBytes,
                token);


            // --------------------------------------------------------
            // ONLY EDITED IMAGE CHANGES
            // --------------------------------------------------------

            SetEditedImage(outputPath);


            AddHistory(
                outputPath,
                "Color Grade",
                settingsJson: JsonSerializer.Serialize(new
                {
                    name = SelectedLut.Name,
                    path = SelectedLut.FilePath,
                    jpeg_quality = 97
                }),
                resourcePath: SelectedLut.FilePath);


            ComparisonPosition = 50;


            StatusMessage =
                "Color grade applied.";
        }
        catch (OperationCanceledException)
        {
            StatusMessage =
                "Processing cancelled.";
        }
        catch (Exception ex)
        {
            StatusMessage =
                $"Color grade failed: {ex.Message}";
        }
        finally
        {
            StopProcessing();
        }
    }


    // ================================================================
    // IMAGE ANALYSIS
    // ================================================================

    [RelayCommand]
    private async Task AnalyzeImageAsync()
    {
        string? sourcePath =
            GetPreviewSourcePath();

        if (string.IsNullOrWhiteSpace(sourcePath) ||
            !File.Exists(sourcePath))
        {
            StatusMessage =
                "Choose an image first.";

            return;
        }

        StartProcessing("Analysing image...");

        try
        {
            _processingCts =
                new CancellationTokenSource();

            CancellationToken token =
                _processingCts.Token;

            string apiInputPath =
                await PrepareApiInputAsync(
                    sourcePath,
                    token);

            byte[] imageBytes =
                await File.ReadAllBytesAsync(
                    apiInputPath,
                    token);

            string jobId =
                Guid.NewGuid().ToString("N");

            _activeJobId =
                jobId;

            AutoEnhanceDataResult result =
                await _api.ExtractImageDataAsync(
                    imageBytes,
                    jobId,
                    OnApiProgress,
                    token);

            if (IsCurrentOriginalImage(sourcePath))
            {
                _analysisJsonForOriginal = result.AnalysisJson;
                _analysisImagePath = sourcePath;
                ApplyRecommendedAutoEnhanceStrength(result.Analysis);
            }

            StatusMessage =
                "Image analysis complete.";
        }
        catch (OperationCanceledException)
        {
            StatusMessage =
                "Processing cancelled.";
        }
        catch (Exception ex)
        {
            StatusMessage =
                $"Image analysis failed: {ex.Message}";
        }
        finally
        {
            StopProcessing();
        }
    }


    // ================================================================
    // MASK DATA
    // ================================================================

    [RelayCommand]
    private async Task ExtractMaskDataAsync()
    {
        string? sourcePath =
            OriginalImagePath;

        if (string.IsNullOrWhiteSpace(sourcePath) ||
            !File.Exists(sourcePath))
        {
            StatusMessage =
                "Choose an image first.";

            return;
        }

        StartProcessing("Analysing image masks...");

        try
        {
            _processingCts =
                new CancellationTokenSource();

            CancellationToken token =
                _processingCts.Token;

            byte[] imageBytes =
                await File.ReadAllBytesAsync(
                    sourcePath,
                    token);

            string jobId =
                Guid.NewGuid().ToString("N");

            _activeJobId =
                jobId;

            MaskDataResult result =
                await _api.MaskDataAsync(
                    imageBytes,
                    jobId,
                    OnApiProgress,
                    token,
                    device: MaskDevice);

            _maskJsonForOriginal =
                result.MaskJsonText;

            _maskImagePath = sourcePath;

            UpdateMaskOptions(
                result.MaskNames,
                result.MaskBinaryPngBase64);

            StatusMessage =
                AvailableMaskNames.Count == 0
                    ? "Mask data is ready."
                    : $"Mask data is ready ({AvailableMaskNames.Count} regions).";
        }
        catch (OperationCanceledException)
        {
            StatusMessage =
                "Processing cancelled.";
        }
        catch (Exception ex)
        {
            StatusMessage =
                $"Mask analysis failed: {ex.Message}";
        }
        finally
        {
            StopProcessing();
        }
    }


    // ================================================================
    // DENOISE
    // ================================================================

    [RelayCommand]
    private Task DenoiseAsync()
    {
        return RunImageOperationAsync(
            "Applying AI denoise...",
            "Denoise",
            "denoise",
            (imageBytes, jobId, token) =>
                _api.DenoiseAsync(
                    imageBytes,
                    device: DenoiseDevice,
                    jobId: jobId,
                    progress: OnApiProgress,
                    cancellationToken: token),
            settingsJson: JsonSerializer.Serialize(new
            {
                device = DenoiseDevice
            }));
    }


    // ================================================================
    // GEOMETRY CORRECTION
    // ================================================================

    [RelayCommand]
    private Task GeometryCorrectionAsync()
    {
        return RunImageOperationAsync(
            "Correcting geometry...",
            "Geometry Correction",
            "geometry",
            (imageBytes, jobId, token) =>
                _api.GeometryCorrectionAsync(
                    imageBytes,
                    mode: GeometryMode,
                    rotate: GeometryRotate,
                    aspect: GeometryAspect,
                    scale: GeometryScale,
                    x: GeometryX,
                    y: GeometryY,
                    crop: GeometryCrop,
                    jobId: jobId,
                    progress: OnApiProgress,
                    cancellationToken: token),
            settingsJson: JsonSerializer.Serialize(new
            {
                mode = GeometryMode,
                rotate = GeometryRotate,
                aspect = GeometryAspect,
                scale = GeometryScale,
                x = GeometryX,
                y = GeometryY,
                crop = GeometryCrop
            }));
    }


    // ================================================================
    // LENS CORRECTION
    // ================================================================

    [RelayCommand]
    private Task LensCorrectionAsync()
    {
        return RunImageOperationAsync(
            "Correcting lens distortion...",
            "Lens Correction",
            "lens",
            (imageBytes, jobId, token) =>
                _api.LensCorrectionAsync(
                    imageBytes,
                    manual: LensManual,
                    k1: LensK1,
                    k2: LensK2,
                    k3: LensK3,
                    p1: LensP1,
                    p2: LensP2,
                    strength: LensStrength,
                    centerX: LensCenterX,
                    centerY: LensCenterY,
                    focalScale: LensFocalScale,
                    noCrop: LensNoCrop,
                    cameraMaker: LensCameraMaker,
                    cameraModel: LensCameraModel,
                    lensMaker: LensMaker,
                    lensModel: LensModel,
                    jobId: jobId,
                    progress: OnApiProgress,
                    cancellationToken: token),
            settingsJson: JsonSerializer.Serialize(new
            {
                manual = LensManual,
                k1 = LensK1,
                k2 = LensK2,
                k3 = LensK3,
                p1 = LensP1,
                p2 = LensP2,
                strength = LensStrength,
                center_x = LensCenterX,
                center_y = LensCenterY,
                focal_scale = LensFocalScale,
                no_crop = LensNoCrop,
                camera_maker = LensCameraMaker,
                camera_model = LensCameraModel,
                lens_maker = LensMaker,
                lens_model = LensModel
            }));
    }


    // ================================================================
    // PIPELINE
    // ================================================================

    [RelayCommand]
    private Task RunPipelineAsync()
    {
        return RunImageOperationAsync(
            "Running pipeline...",
            "Pipeline",
            "pipeline",
            (imageBytes, jobId, token) =>
                _api.PipelineAsync(
                    imageBytes,
                    stages: PipelineStages,
                    jobId: jobId,
                    progress: OnApiProgress,
                    cancellationToken: token),
            settingsJson: JsonSerializer.Serialize(new
            {
                stages = PipelineStages
             }));
    }


    // ================================================================
    // API HEALTH
    // ================================================================

    [RelayCommand]
    private async Task CheckApiHealthAsync()
    {
        try
        {
            StatusMessage =
                await _api.IsHealthyAsync()
                    ? "API is connected."
                    : "API is unavailable.";
        }
        catch (Exception ex)
        {
            StatusMessage =
                $"API health check failed: {ex.Message}";
        }
    }


    // ================================================================
    // SHARED IMAGE OPERATION
    // ================================================================

    private async Task RunImageOperationAsync(
        string startMessage,
        string operation,
        string fileSuffix,
        Func<byte[], string, CancellationToken, Task<AutoEnhanceApiResult>> operationCall,
        string? settingsJson = null,
        string? analysisJson = null,
        string? maskJson = null,
        IReadOnlyCollection<string>? maskNames = null,
        bool isMaskScoped = false,
        string? resourcePath = null)
    {
        string? sourcePath = GetPreviewSourcePath();

        if (string.IsNullOrWhiteSpace(sourcePath) ||
            !File.Exists(sourcePath))
        {
            StatusMessage =
                "Choose an image first.";

            return;
        }

        StartProcessing(startMessage);

        try
        {
            _processingCts =
                new CancellationTokenSource();

            CancellationToken token =
                _processingCts.Token;

            string apiInputPath = await PrepareApiInputAsync(
                sourcePath,
                token);

            byte[] imageBytes =
                await File.ReadAllBytesAsync(
                    apiInputPath,
                    token);

            string jobId =
                Guid.NewGuid().ToString("N");

            _activeJobId =
                jobId;

            AutoEnhanceApiResult result =
                await operationCall(
                    imageBytes,
                    jobId,
                    token);

            token.ThrowIfCancellationRequested();

            string outputDirectory =
                Path.Combine(
                    Path.GetTempPath(),
                    "AutoPhotoEditor",
                    "Generated");

            Directory.CreateDirectory(
                outputDirectory);

            string outputPath =
                Path.Combine(
                    outputDirectory,
                    $"{jobId}_{fileSuffix}.png");

            await File.WriteAllBytesAsync(
                outputPath,
                result.ImageBytes,
                token);

            SetEditedImage(outputPath);

            AddHistory(
                outputPath,
                operation,
                maskJson: maskJson,
                settingsJson: settingsJson,
                analysisJson: analysisJson,
                maskNames: maskNames,
                isMaskScoped: isMaskScoped,
                resourcePath: resourcePath);

            ComparisonPosition =
                50;

            StatusMessage =
                $"{operation} complete.";
        }
        catch (OperationCanceledException)
        {
            StatusMessage =
                "Processing cancelled.";
        }
        catch (Exception ex)
        {
            StatusMessage =
                $"{operation} failed: {ex.Message}";
        }
        finally
        {
            StopProcessing();
        }
    }


    private void UpdateMaskOptions(
        IEnumerable<string> maskNames,
        IReadOnlyDictionary<string, string>? binaryPngs = null)
    {
        foreach (MaskOption option in MaskOptions)
            option.PropertyChanged -= MaskOption_PropertyChanged;

        MaskOptions.Clear();
        AvailableMaskNames.Clear();

        foreach (string maskName in maskNames)
        {
            if (string.IsNullOrWhiteSpace(maskName) ||
                MaskOptions.Any(option => string.Equals(option.Name, maskName, StringComparison.OrdinalIgnoreCase)))
            {
                continue;
            }

            string? binaryPng = null;
            binaryPngs?.TryGetValue(maskName, out binaryPng);
            var option = new MaskOption(maskName, binaryPng);
            option.PropertyChanged += MaskOption_PropertyChanged;
            MaskOptions.Add(option);
            AvailableMaskNames.Add(maskName);
        }

        OnPropertyChanged(nameof(HasAvailableMasks));
        OnPropertyChanged(nameof(MaskSelectionHint));
    }

    [RelayCommand]
    private void SelectAllMasks()
    {
        foreach (MaskOption option in MaskOptions)
            option.IsSelected = true;
    }

    [RelayCommand]
    private void ClearMaskSelection()
    {
        foreach (MaskOption option in MaskOptions)
            option.IsSelected = false;

        SelectedMaskNamesText = string.Empty;
        SelectedMaskName = null;
    }

    private void UpdateMaskOptions(JsonElement maskJson)
    {
        if (maskJson.ValueKind != JsonValueKind.Object ||
            !maskJson.TryGetProperty("masks", out JsonElement masks) ||
            masks.ValueKind != JsonValueKind.Object)
        {
            UpdateMaskOptions(Array.Empty<string>());
            return;
        }

        var names = new List<string>();
        var binaryPngs = new Dictionary<string, string>();
        foreach (JsonProperty mask in masks.EnumerateObject())
        {
            names.Add(mask.Name);
            if (mask.Value.TryGetProperty("binary_png_base64", out JsonElement binary) &&
                binary.ValueKind == JsonValueKind.String)
            {
                binaryPngs[mask.Name] = binary.GetString() ?? string.Empty;
            }
        }

        UpdateMaskOptions(names, binaryPngs);
    }

    private void MaskOption_PropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
    {
        if (e.PropertyName != nameof(MaskOption.IsSelected))
            return;

        string[] selectedNames = MaskOptions
            .Where(option => option.IsSelected)
            .Select(option => option.Name)
            .ToArray();

        SelectedMaskNamesText = string.Join(", ", selectedNames);
        SelectedMaskName = selectedNames.FirstOrDefault();
        if (SelectedMaskName != null)
            ApplySelectedMaskRecommendation(SelectedMaskName);
    }

    private void RemoveEnhancedMasks(IEnumerable<string> maskNames)
    {
        var enhancedNames = new HashSet<string>(maskNames, StringComparer.OrdinalIgnoreCase);
        foreach (MaskOption option in MaskOptions
                     .Where(option => enhancedNames.Contains(option.Name))
                     .ToList())
        {
            option.PropertyChanged -= MaskOption_PropertyChanged;
            MaskOptions.Remove(option);
            AvailableMaskNames.Remove(option.Name);
        }

        SelectedMaskNamesText = string.Empty;
        SelectedMaskName = null;
        OnPropertyChanged(nameof(HasAvailableMasks));
        OnPropertyChanged(nameof(MaskSelectionHint));
    }


    private static IReadOnlyList<string> GetMaskNames(
        JsonElement maskJson)
    {
        var result =
            new List<string>();

        if (maskJson.ValueKind != JsonValueKind.Object ||
            !maskJson.TryGetProperty(
                "masks",
                out JsonElement masks) ||
            masks.ValueKind != JsonValueKind.Object)
        {
            return result;
        }

        foreach (JsonProperty mask in masks.EnumerateObject())
        {
            result.Add(
                mask.Name);
        }

        return result;
    }


    private IReadOnlyCollection<string>? ParseSelectedMaskNames()
    {
        string[] checkedNames = MaskOptions
            .Where(option => option.IsSelected)
            .Select(option => option.Name)
            .ToArray();

        if (checkedNames.Length > 0)
            return checkedNames;

        if (string.IsNullOrWhiteSpace(SelectedMaskNamesText))
        {
            return string.IsNullOrWhiteSpace(SelectedMaskName)
                ? null
                : new[] { SelectedMaskName };
        }

        string[] names =
            SelectedMaskNamesText
                .Split(
                    new[] { ',', ';', '\n', '\r' },
                    StringSplitOptions.RemoveEmptyEntries |
                    StringSplitOptions.TrimEntries)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray();

        if (AvailableMaskNames.Count > 0)
        {
            var availableNames =
                AvailableMaskNames.ToHashSet(StringComparer.OrdinalIgnoreCase);

            names = names
                .Where(availableNames.Contains)
                .ToArray();
        }

        return names.Length == 0
            ? null
            : names;
    }

    private string? GetReusableMaskJson()
    {
        if (!string.IsNullOrWhiteSpace(_maskJsonForOriginal) &&
            string.Equals(
                _maskImagePath,
                OriginalImagePath,
                StringComparison.OrdinalIgnoreCase))
        {
            return _maskJsonForOriginal;
        }

        if (string.IsNullOrWhiteSpace(OriginalImagePath))
            return null;

        return JobHistory
            .LastOrDefault(
                item =>
                    item.Operation == PhotoEditOperation.MaskedEnhance &&
                    !string.IsNullOrWhiteSpace(item.MaskJson) &&
                    string.Equals(
                        item.SourceImagePath,
                        OriginalImagePath,
                        StringComparison.OrdinalIgnoreCase))
            ?.MaskJson;
    }


    // ================================================================
    // SET EDITED IMAGE
    // ================================================================

    private void SetEditedImage(string path)
    {
        if (!File.Exists(path))
            return;


        EditedImagePath = path;


        EditedImage =
            LoadDisplayPreviewBitmap(
                path,
                DisplayPreviewMaxDimension);
    }

    private void SetLivePreviewImage(string path)
    {
        if (!File.Exists(path))
            return;

        EditedImagePath = path;
        EditedImage = LoadDisplayPreviewBitmap(
            path,
            DisplayPreviewMaxDimension);
    }

    private string? GetPreviewSourcePath()
    {
        if (HistoryIndex >= 0 &&
            HistoryIndex < JobHistory.Count)
        {
            string path = JobHistory[HistoryIndex].OutputImagePath;
            if (!string.IsNullOrWhiteSpace(path) && File.Exists(path))
                return path;
        }

        return OriginalImagePath;
    }


    // ================================================================
    // CANCEL
    // ================================================================

    [RelayCommand]
    private async Task CancelProcessingAsync()
    {
        if (_processingCts == null &&
            string.IsNullOrWhiteSpace(_activeJobId))
        {
            return;
        }


        StatusMessage =
            "Cancelling...";

        string? jobId =
            _activeJobId;

        _processingCts?.Cancel();

        try
        {
            if (!string.IsNullOrWhiteSpace(jobId))
            {
                try
                {
                    await _api.CancelJobAsync(
                        jobId);
                }
                catch
                {
                    // The local cancellation still stops the active request.
                }
            }
        }
        finally
        {
            StopProcessing();
        }
    }


    // ================================================================
    // HISTORY NAVIGATION
    // ================================================================

    [RelayCommand]
    private void PreviousEdit()
    {
        if (HistoryIndex < 0)
        {
            return;
        }

        HistoryIndex--;
        ApplyHistoryState(
            HistoryIndex);

        StatusMessage = "Showing previous edit.";
    }

    [RelayCommand]
    private void NextEdit()
    {
        if (HistoryIndex >= JobHistory.Count - 1)
            return;


        HistoryIndex++;


        ApplyHistoryState(
            HistoryIndex);

        StatusMessage = "Showing next edit.";
    }


    // ================================================================
    // SAVE IMAGE
    // ================================================================

    [RelayCommand]
    private async Task SaveImageAsync()
    {
        string? originalPath = OriginalImagePath;

        if (string.IsNullOrWhiteSpace(originalPath) ||
            !File.Exists(originalPath))
        {
            StatusMessage =
                "There is no image to export.";

            return;
        }


        var dialog = new SaveFileDialog
        {
            Title = "Export image",
            Filter =
                "JPEG Image|*.jpg;*.jpeg|PNG Image|*.png",
            FileName = "edited-image.jpg"
        };


        if (dialog.ShowDialog() != true)
            return;

        IReadOnlyList<PhotoEditJob> activeEdits =
            HistoryIndex < 0
                ? Array.Empty<PhotoEditJob>()
                : JobHistory
                    .Take(Math.Min(HistoryIndex + 1, JobHistory.Count))
                    .OrderBy(edit => edit.Sequence)
                    .ThenBy(edit => JobHistory.IndexOf(edit))
                    .ToArray();

        StartProcessing("Preparing full-resolution export...");

        try
        {
            _processingCts = new CancellationTokenSource();
            CancellationToken token = _processingCts.Token;

            byte[] imageBytes =
                await File.ReadAllBytesAsync(originalPath, token);

            if (imageBytes.Length == 0)
                throw new InvalidDataException("The original image is empty.");

            byte[] exportedBytes =
                await ReplayHistoryAsync(imageBytes, activeEdits, token);

            token.ThrowIfCancellationRequested();
            ExportImage(exportedBytes, dialog.FileName);

            PhotoEditJob? maskedEdit = activeEdits
                .LastOrDefault(edit =>
                    !string.IsNullOrWhiteSpace(edit.MaskJson));

            string? maskJson = maskedEdit?.MaskJson;

            if (!string.IsNullOrWhiteSpace(maskJson))
            {
                string directory =
                    Path.GetDirectoryName(dialog.FileName)
                    ?? Environment.CurrentDirectory;

                string maskPath =
                    Path.Combine(
                        directory,
                        Path.GetFileNameWithoutExtension(dialog.FileName) +
                        ".masks.json");

                File.WriteAllText(
                    maskPath,
                    maskJson);

                StatusMessage =
                    "Image and mask data exported successfully.";
            }
            else
            {
                StatusMessage =
                    "Image exported successfully.";
            }
        }
        catch (Exception ex)
        {
            StatusMessage =
                $"Export failed: {ex.Message}";
        }
        finally
        {
            StopProcessing();
        }
    }

    private async Task<byte[]> ReplayHistoryAsync(
        byte[] imageBytes,
        IReadOnlyList<PhotoEditJob> edits,
        CancellationToken token)
    {
        byte[] currentBytes = imageBytes;

        for (int index = 0; index < edits.Count; index++)
        {
            PhotoEditJob edit = edits[index];
            token.ThrowIfCancellationRequested();

            StatusMessage =
                $"Exporting edit {index + 1} of {edits.Count}: {GetOperationDisplayName(edit.Operation)}...";

            currentBytes =
                await ApplyHistoryEditAsync(currentBytes, edit, token);
        }

        return currentBytes;
    }

    private async Task<byte[]> ApplyHistoryEditAsync(
        byte[] imageBytes,
        PhotoEditJob edit,
        CancellationToken token)
    {
        switch (edit.Operation)
        {
            case PhotoEditOperation.AutoEnhance:
            {
                JsonElement settings = ParseHistorySettings(edit.SettingsJson);
                double strength = Math.Clamp(
                    GetHistoryNumber(settings, "strength", 1.0),
                    0.0,
                    1.0);

                string analysisJobId = Guid.NewGuid().ToString("N");
                _activeJobId = analysisJobId;
                AutoEnhanceDataResult analysis =
                    await _api.ExtractImageDataAsync(
                        imageBytes,
                        analysisJobId,
                        OnApiProgress,
                        token,
                        summaryOnly: false);

                token.ThrowIfCancellationRequested();

                string jobId = Guid.NewGuid().ToString("N");
                _activeJobId = jobId;
                AutoEnhanceApiResult result =
                    await _api.AutoEnhanceAsync(
                        imageBytes,
                        analysis.Analysis,
                        strength,
                        jobId,
                        OnApiProgress,
                        token);

                return result.ImageBytes;
            }

            case PhotoEditOperation.MaskedEnhance:
            {
                JsonElement settings = ParseHistorySettings(edit.SettingsJson);
                JsonElement maskJson = ParseHistoryMask(edit);
                IReadOnlyCollection<string> maskNames =
                    GetHistoryMaskNames(edit, settings);
                double strength = Math.Clamp(
                    GetHistoryNumber(settings, "strength", 1.0),
                    0.0,
                    1.0);
                double feather = Math.Max(
                    0.0,
                    GetHistoryNumber(settings, "feather", 2.0));

                string jobId = Guid.NewGuid().ToString("N");
                _activeJobId = jobId;
                AutoEnhanceApiResult result =
                    await _api.MaskedEnhanceAsync(
                        imageBytes,
                        maskJson,
                        strength,
                        feather,
                        maskNames,
                        jobId,
                        OnApiProgress,
                        token);

                return result.ImageBytes;
            }

            case PhotoEditOperation.ManualAdjust:
            {
                JsonElement settings = ParseHistorySettings(edit.SettingsJson);
                JsonElement adjustments =
                    TryGetHistoryProperty(settings, "adjustments", out JsonElement nestedAdjustments) &&
                    nestedAdjustments.ValueKind == JsonValueKind.Object
                        ? nestedAdjustments
                        : settings;

                var manualSettings = new ManualAdjustmentSettings
                {
                    Exposure = GetHistoryNumber(adjustments, "exposure", 0.0),
                    Contrast = GetHistoryNumber(adjustments, "contrast", 0.0),
                    Highlights = GetHistoryNumber(adjustments, "highlights", 0.0),
                    Shadows = GetHistoryNumber(adjustments, "shadows", 0.0),
                    Temperature = GetHistoryNumber(adjustments, "temperature", 0.0),
                    Tint = GetHistoryNumber(adjustments, "tint", 0.0),
                    Saturation = GetHistoryNumber(adjustments, "saturation", 0.0),
                    Sharpness = GetHistoryNumber(adjustments, "sharpness", 0.0),
                    UseMask = edit.IsMaskScoped,
                    MaskNames = GetHistoryMaskNames(edit, settings),
                    Feather = Math.Max(
                        0.0,
                        GetHistoryNumber(settings, "feather", 2.0))
                };

                if (edit.IsMaskScoped)
                    manualSettings.MaskJson = ParseHistoryMask(edit);

                string jobId = Guid.NewGuid().ToString("N");
                _activeJobId = jobId;
                AutoEnhanceApiResult result =
                    await _api.ManualAdjustAsync(
                        imageBytes,
                        manualSettings,
                        jobId,
                        OnApiProgress,
                        token);

                return result.ImageBytes;
            }

            case PhotoEditOperation.Denoise:
            {
                JsonElement settings = ParseHistorySettings(edit.SettingsJson);
                string device = GetHistoryString(settings, "device") ?? "cpu";
                string jobId = Guid.NewGuid().ToString("N");
                _activeJobId = jobId;

                AutoEnhanceApiResult result =
                    await _api.DenoiseAsync(
                        imageBytes,
                        device,
                        jobId,
                        OnApiProgress,
                        token);

                return result.ImageBytes;
            }

            case PhotoEditOperation.GeometryCorrection:
            {
                JsonElement settings = ParseHistorySettings(edit.SettingsJson);
                string mode = GetHistoryString(settings, "mode") ?? "auto";
                double rotate = GetHistoryNumber(settings, "rotate", 0.0);
                double aspect = GetHistoryNumber(settings, "aspect", 0.0);
                double scale = Math.Max(
                    0.01,
                    GetHistoryNumber(settings, "scale", 100.0));
                double x = GetHistoryNumber(settings, "x", 0.0);
                double y = GetHistoryNumber(settings, "y", 0.0);
                bool crop = GetHistoryBoolean(settings, "crop", true);
                string jobId = Guid.NewGuid().ToString("N");
                _activeJobId = jobId;

                AutoEnhanceApiResult result =
                    await _api.GeometryCorrectionAsync(
                        imageBytes,
                        mode,
                        rotate,
                        aspect: aspect,
                        scale: scale,
                        x: x,
                        y: y,
                        crop: crop,
                        jobId: jobId,
                        progress: OnApiProgress,
                        cancellationToken: token);

                return result.ImageBytes;
            }

            case PhotoEditOperation.LensCorrection:
            {
                JsonElement settings = ParseHistorySettings(edit.SettingsJson);
                string jobId = Guid.NewGuid().ToString("N");
                _activeJobId = jobId;

                AutoEnhanceApiResult result =
                    await _api.LensCorrectionAsync(
                        imageBytes,
                        manual: GetHistoryBoolean(settings, "manual", false),
                        k1: GetHistoryNumber(settings, "k1", 0.0),
                        k2: GetHistoryNumber(settings, "k2", 0.0),
                        k3: GetHistoryNumber(settings, "k3", 0.0),
                        p1: GetHistoryNumber(settings, "p1", 0.0),
                        p2: GetHistoryNumber(settings, "p2", 0.0),
                        strength: GetHistoryNumber(settings, "strength", 1.0),
                        centerX: GetHistoryNumber(settings, "center_x", 0.5),
                        centerY: GetHistoryNumber(settings, "center_y", 0.5),
                        focalScale: GetHistoryNumber(settings, "focal_scale", 1.0),
                        noCrop: GetHistoryBoolean(settings, "no_crop", false),
                        cameraMaker: GetHistoryString(settings, "camera_maker"),
                        cameraModel: GetHistoryString(settings, "camera_model"),
                        lensMaker: GetHistoryString(settings, "lens_maker"),
                        lensModel: GetHistoryString(settings, "lens_model"),
                        jobId: jobId,
                        progress: OnApiProgress,
                        cancellationToken: token);

                return result.ImageBytes;
            }

            case PhotoEditOperation.ColorGrade:
            {
                JsonElement settings = ParseHistorySettings(edit.SettingsJson);
                string? preset = edit.ResourcePath ??
                    GetHistoryString(settings, "path");

                if (string.IsNullOrWhiteSpace(preset))
                {
                    throw new InvalidOperationException(
                        "The color grade edit does not contain its LUT path.");
                }

                int jpegQuality = Math.Clamp(
                    (int)Math.Round(GetHistoryNumber(settings, "jpeg_quality", 97.0)),
                    1,
                    100);
                string jobId = Guid.NewGuid().ToString("N");
                _activeJobId = jobId;

                AutoEnhanceApiResult result =
                    await _api.ApplyPresetAsync(
                        imageBytes,
                        preset,
                        jpegQuality,
                        jobId,
                        OnApiProgress,
                        token);

                return result.ImageBytes;
            }

            case PhotoEditOperation.Pipeline:
            {
                JsonElement settings = ParseHistorySettings(edit.SettingsJson);
                string stages = GetHistoryString(settings, "stages") ??
                    "analyze,auto-enhance";
                string jobId = Guid.NewGuid().ToString("N");
                _activeJobId = jobId;

                AutoEnhanceApiResult result =
                    await _api.PipelineAsync(
                        imageBytes,
                        stages,
                        jobId,
                        OnApiProgress,
                        token);

                return result.ImageBytes;
            }

            default:
                throw new NotSupportedException(
                    $"The edit type '{edit.Operation}' cannot be replayed for export.");
        }
    }

    private static string GetOperationDisplayName(PhotoEditOperation operation)
    {
        return operation switch
        {
            PhotoEditOperation.AutoEnhance => "Auto Enhance",
            PhotoEditOperation.MaskedEnhance => "Masked Enhance",
            PhotoEditOperation.ManualAdjust => "Manual Adjust",
            PhotoEditOperation.Denoise => "Denoise",
            PhotoEditOperation.GeometryCorrection => "Geometry Correction",
            PhotoEditOperation.LensCorrection => "Lens Correction",
            PhotoEditOperation.ColorGrade => "Color Grade",
            PhotoEditOperation.Pipeline => "Pipeline",
            _ => operation.ToString()
        };
    }

    private static JsonElement ParseHistorySettings(string? settingsJson)
    {
        if (string.IsNullOrWhiteSpace(settingsJson))
            return default;

        try
        {
            using JsonDocument document = JsonDocument.Parse(settingsJson);
            return document.RootElement.Clone();
        }
        catch (JsonException ex)
        {
            throw new InvalidOperationException(
                "An edit contains invalid settings JSON.",
                ex);
        }
    }

    private static JsonElement ParseHistoryMask(PhotoEditJob edit)
    {
        if (string.IsNullOrWhiteSpace(edit.MaskJson))
        {
            throw new InvalidOperationException(
                $"The {GetOperationDisplayName(edit.Operation)} edit does not contain mask data.");
        }

        try
        {
            using JsonDocument document = JsonDocument.Parse(edit.MaskJson);
            if (document.RootElement.ValueKind != JsonValueKind.Object)
            {
                throw new InvalidOperationException(
                    "The edit contains invalid mask data.");
            }

            return document.RootElement.Clone();
        }
        catch (JsonException ex)
        {
            throw new InvalidOperationException(
                "The edit contains invalid mask JSON.",
                ex);
        }
    }

    private static IReadOnlyList<string> GetHistoryMaskNames(
        PhotoEditJob edit,
        JsonElement settings)
    {
        if (edit.MaskNames.Count > 0)
            return edit.MaskNames;

        if (!TryGetHistoryProperty(settings, "mask_names", out JsonElement names) ||
            names.ValueKind != JsonValueKind.Array)
        {
            return Array.Empty<string>();
        }

        return names
            .EnumerateArray()
            .Where(item => item.ValueKind == JsonValueKind.String)
            .Select(item => item.GetString())
            .Where(name => !string.IsNullOrWhiteSpace(name))
            .Select(name => name!.Trim())
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    private static bool TryGetHistoryProperty(
        JsonElement element,
        string propertyName,
        out JsonElement value)
    {
        value = default;
        if (element.ValueKind != JsonValueKind.Object)
            return false;

        if (element.TryGetProperty(propertyName, out value))
            return true;

        foreach (JsonProperty property in element.EnumerateObject())
        {
            if (string.Equals(
                    property.Name,
                    propertyName,
                    StringComparison.OrdinalIgnoreCase))
            {
                value = property.Value;
                return true;
            }
        }

        return false;
    }

    private static double GetHistoryNumber(
        JsonElement settings,
        string propertyName,
        double fallback)
    {
        return TryGetHistoryProperty(settings, propertyName, out JsonElement value) &&
               value.ValueKind == JsonValueKind.Number &&
               value.TryGetDouble(out double number) &&
               double.IsFinite(number)
            ? number
            : fallback;
    }

    private static string? GetHistoryString(
        JsonElement settings,
        string propertyName)
    {
        return TryGetHistoryProperty(settings, propertyName, out JsonElement value) &&
               value.ValueKind == JsonValueKind.String
            ? value.GetString()
            : null;
    }

    private static bool GetHistoryBoolean(
        JsonElement settings,
        string propertyName,
        bool fallback)
    {
        return TryGetHistoryProperty(settings, propertyName, out JsonElement value) &&
               (value.ValueKind == JsonValueKind.True ||
                value.ValueKind == JsonValueKind.False)
            ? value.GetBoolean()
            : fallback;
    }

    private static void ExportImage(
        byte[] imageBytes,
        string destinationPath)
    {
        using var input = new MemoryStream(imageBytes, writable: false);

        BitmapDecoder decoder =
            BitmapDecoder.Create(
                input,
                BitmapCreateOptions.PreservePixelFormat,
                BitmapCacheOption.OnLoad);

        BitmapSource source = decoder.Frames[0];
        BitmapEncoder encoder =
            Path.GetExtension(destinationPath).ToLowerInvariant() switch
            {
                ".jpg" or ".jpeg" => new JpegBitmapEncoder
                {
                    QualityLevel = 97
                },
                ".bmp" => new BmpBitmapEncoder(),
                ".tif" or ".tiff" => new TiffBitmapEncoder(),
                _ => new PngBitmapEncoder()
            };

        encoder.Frames.Add(BitmapFrame.Create(source));

        using FileStream output =
            new(
                destinationPath,
                FileMode.Create,
                FileAccess.Write,
                FileShare.None);

        encoder.Save(output);
    }


    // ================================================================
    // CURRENT IMAGE PATH
    // ================================================================

    private PhotoEditJob? GetCurrentHistoryJob()
    {
        if (HistoryIndex < 0 ||
            HistoryIndex >= JobHistory.Count)
        {
            return null;
        }

        return JobHistory[HistoryIndex];
    }


    // ================================================================
    // PREPARE API INPUT
    // ================================================================

    private async Task<string> PrepareApiInputAsync(
        string sourcePath,
        CancellationToken token)
    {
        // ------------------------------------------------------------
        // FULL ORIGINAL MODE
        //
        // Send original file directly.
        // ------------------------------------------------------------

        int maxDimension =
            Math.Clamp(
                PreviewMaxDimension,
                320,
                8000);


        int quality =
            Math.Clamp(
                PreviewJpegQuality,
                1,
                100);


        string tempDirectory =
            Path.Combine(
                Path.GetTempPath(),
                "AutoPhotoEditor",
                "PreviewApi");


        Directory.CreateDirectory(
            tempDirectory);


        string tempPath =
            Path.Combine(
                tempDirectory,
                $"{Guid.NewGuid():N}.jpg");


        await Task.Run(
            () =>
            {
                token.ThrowIfCancellationRequested();


                using FileStream stream =
                    new(
                        sourcePath,
                        FileMode.Open,
                        FileAccess.Read,
                        FileShare.Read);


                BitmapDecoder decoder =
                    BitmapDecoder.Create(
                        stream,
                        BitmapCreateOptions.PreservePixelFormat,
                        BitmapCacheOption.OnLoad);


                BitmapSource source =
                    decoder.Frames[0];


                int width =
                    source.PixelWidth;

                int height =
                    source.PixelHeight;


                double scale =
                    Math.Min(
                        1.0,
                        Math.Min(
                            (double)maxDimension / width,
                            (double)maxDimension / height));


                BitmapSource bitmap =
                    source;


                if (scale < 1.0)
                {
                    var transformed =
                        new TransformedBitmap(
                            source,
                            new ScaleTransform(
                                scale,
                                scale));


                    transformed.Freeze();


                    bitmap =
                        transformed;
                }


                var encoder =
                    new JpegBitmapEncoder
                    {
                        QualityLevel = quality
                    };


                encoder.Frames.Add(
                    BitmapFrame.Create(bitmap));


                using FileStream output =
                    new(
                        tempPath,
                        FileMode.Create,
                        FileAccess.Write,
                        FileShare.None);


                encoder.Save(output);
            },
            token);


        return tempPath;
    }


    // ================================================================
    // DISPLAY PREVIEW LOADER
    // ================================================================

    private static BitmapImage LoadDisplayPreviewBitmap(
        string path,
        int maxDimension)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException(
                "Image file was not found.",
                path);
        }


        maxDimension =
            Math.Clamp(
                maxDimension,
                320,
                4000);


        using FileStream stream =
            new(
                path,
                FileMode.Open,
                FileAccess.Read,
                FileShare.Read);


        BitmapDecoder decoder =
            BitmapDecoder.Create(
                stream,
                BitmapCreateOptions.PreservePixelFormat,
                BitmapCacheOption.OnLoad);


        BitmapFrame frame =
            decoder.Frames[0];


        int sourceWidth =
            frame.PixelWidth;


        int sourceHeight =
            frame.PixelHeight;


        double scale =
            Math.Min(
                1.0,
                Math.Min(
                    (double)maxDimension / sourceWidth,
                    (double)maxDimension / sourceHeight));


        BitmapSource source =
            frame;


        if (scale < 1.0)
        {
            int targetWidth =
                Math.Max(
                    1,
                    (int)Math.Round(
                        sourceWidth * scale));


            int targetHeight =
                Math.Max(
                    1,
                    (int)Math.Round(
                        sourceHeight * scale));


            var resized =
                new TransformedBitmap(
                    frame,
                    new ScaleTransform(
                        (double)targetWidth / sourceWidth,
                        (double)targetHeight / sourceHeight));


            resized.Freeze();


            source =
                resized;
        }


        byte[] png =
            EncodePreviewToPng(source);


        using var memory =
            new MemoryStream(png);


        var bitmap =
            new BitmapImage();


        bitmap.BeginInit();


        bitmap.CacheOption =
            BitmapCacheOption.OnLoad;


        bitmap.CreateOptions =
            BitmapCreateOptions.PreservePixelFormat;


        bitmap.StreamSource =
            memory;


        bitmap.EndInit();


        bitmap.Freeze();


        return bitmap;
    }


    // ================================================================
    // PNG ENCODER
    // ================================================================

    private static byte[] EncodePreviewToPng(
        BitmapSource source)
    {
        var encoder =
            new PngBitmapEncoder();


        encoder.Frames.Add(
            BitmapFrame.Create(source));


        using var stream =
            new MemoryStream();


        encoder.Save(stream);


        return stream.ToArray();
    }


    // ================================================================
    // HISTORY
    // ================================================================

    private void AddHistory(
        string outputPath,
        string operation,
        string? maskJson = null,
        string? settingsJson = null,
        string? analysisJson = null,
        IReadOnlyCollection<string>? maskNames = null,
        bool isMaskScoped = false,
        string? resourcePath = null)
    {
        string sourcePath =
            OriginalImagePath ?? string.Empty;

        string? parentJobId =
            GetCurrentHistoryJob()?.JobId;

        // ------------------------------------------------------------
        // If we previously undid something, remove the abandoned
        // redo states before adding a new edit.
        // ------------------------------------------------------------

        while (JobHistory.Count - 1 > HistoryIndex)
        {
            PhotoEditJob abandoned =
                JobHistory[^1];

            JobHistory.RemoveAt(
                JobHistory.Count - 1);

            DeleteGeneratedFileIfSafe(
                abandoned.OutputImagePath);
        }

        string jobId =
            Path.GetFileNameWithoutExtension(outputPath)
                .Split('_', 2, StringSplitOptions.RemoveEmptyEntries)
                .FirstOrDefault() ??
            Guid.NewGuid().ToString("N");

        PhotoEditOperation editOperation =
            operation switch
            {
                "Auto Enhance" => PhotoEditOperation.AutoEnhance,
                "Masked Enhance" => PhotoEditOperation.MaskedEnhance,
                "Manual Adjust" => PhotoEditOperation.ManualAdjust,
                "Denoise" => PhotoEditOperation.Denoise,
                "Geometry Correction" => PhotoEditOperation.GeometryCorrection,
                "Lens Correction" => PhotoEditOperation.LensCorrection,
                "Color Grade" => PhotoEditOperation.ColorGrade,
                "Pipeline" => PhotoEditOperation.Pipeline,
                _ => PhotoEditOperation.Other
            };

        JobHistory.Add(
            new PhotoEditJob
            {
                JobId = jobId,
                Sequence = JobHistory.Count + 1,
                OutputImagePath = outputPath,
                SourceImagePath = sourcePath,
                ParentJobId = parentJobId,
                MaskJson = maskJson,
                SettingsJson = settingsJson,
                AnalysisJson = analysisJson,
                ResourcePath = resourcePath,
                MaskNames = maskNames?.ToArray() ?? Array.Empty<string>(),
                IsMaskScoped = isMaskScoped,
                Operation = editOperation
            });

        HistoryIndex =
            JobHistory.Count - 1;

        RefreshHistoryState();
    }

    private static void DeleteGeneratedFileIfSafe(string path)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            return;

        string generatedDirectory =
            Path.GetFullPath(
                Path.Combine(
                    Path.GetTempPath(),
                    "AutoPhotoEditor",
                    "Generated"));

        string fullPath = Path.GetFullPath(path);

        if (!fullPath.StartsWith(
                generatedDirectory + Path.DirectorySeparatorChar,
                StringComparison.OrdinalIgnoreCase))
        {
            return;
        }

        try
        {
            File.Delete(fullPath);
        }
        catch
        {
            // An abandoned state can remain on disk without affecting the timeline.
        }
    }

    private void RefreshHistoryState()
    {
        OnPropertyChanged(nameof(CanGoPrevious));
        OnPropertyChanged(nameof(CanGoNext));
        OnPropertyChanged(nameof(HistoryPositionText));
    }


    // ================================================================
    // APPLY HISTORY
    // ================================================================

    private void ApplyHistoryState(int index)
    {
        // ------------------------------------------------------------
        // IMPORTANT:
        // index < 0 means "original image".
        // ------------------------------------------------------------

        if (index < 0)
        {
            // Explicitly switch back to the original image state.
            EditedImagePath = null;
            EditedImage = null;

            ComparisonPosition = 50;

            ResetAdjustmentControls();

            return;
        }


        if (index >= JobHistory.Count)
            return;


        string path =
            JobHistory[index].OutputImagePath;


        if (!File.Exists(path))
            return;


        SetEditedImage(path);

        ApplyHistoryControls(index);

        // Keep selected history states centered for before/after comparison.
        ComparisonPosition = 50;
    }

    private void ApplyHistoryControls(int index)
    {
        if (index < 0 || index >= JobHistory.Count)
        {
            ResetAdjustmentControls();
            return;
        }

        PhotoEditJob job = JobHistory[index];

        ResetManualControls();

        if (job.Operation == PhotoEditOperation.AutoEnhance &&
            TryGetNumber(job.SettingsJson, "strength", out double autoStrength))
        {
            AutoEnhanceStrength = Math.Clamp(autoStrength, 0.0, 1.0);
            ApplyAutoAnalysisSettings(job.AnalysisJson);
        }

        if (job.Operation == PhotoEditOperation.MaskedEnhance)
        {
            if (TryGetNumber(job.SettingsJson, "strength", out double maskedStrength))
                MaskedStrength = Math.Clamp(maskedStrength, 0.0, 1.0);

            if (TryGetNumber(job.SettingsJson, "feather", out double feather))
                MaskedFeather = Math.Max(0.0, feather);

            SelectedMaskNamesText = string.Join(", ", job.MaskNames);
            SelectedMaskName = job.MaskNames.FirstOrDefault();
            ApplyMaskedReportSettings(job.SettingsJson);
        }

        if (job.Operation == PhotoEditOperation.ManualAdjust)
        {
            ApplyManualSettings(job.SettingsJson);
            ManualAdjustMask = job.IsMaskScoped;
            SelectedMaskNamesText = string.Join(", ", job.MaskNames);
            SelectedMaskName = job.MaskNames.FirstOrDefault();
        }
    }

    private void ResetAdjustmentControls()
    {
        AutoEnhanceStrength = 1.0;
        MaskedStrength = 1.0;
        MaskedFeather = 2.0;
        ResetManualControls();
        SelectedMaskNamesText = string.Empty;
        SelectedMaskName = null;
    }

    private void ResetManualControls()
    {
        ManualExposure = 0.0;
        ManualContrast = 0.0;
        ManualHighlights = 0.0;
        ManualShadows = 0.0;
        ManualTemperature = 0.0;
        ManualTint = 0.0;
        ManualSaturation = 0.0;
        ManualSharpness = 0.0;
        ManualAdjustMask = false;
    }

    private void ApplyAutoAnalysisSettings(string? analysisJson)
    {
        if (string.IsNullOrWhiteSpace(analysisJson))
            return;

        try
        {
            using JsonDocument document = JsonDocument.Parse(analysisJson);

            JsonElement root = document.RootElement;
            JsonElement recommendations = root.TryGetProperty("recommendations", out JsonElement recommendationValue) &&
                                           recommendationValue.ValueKind == JsonValueKind.Object
                ? recommendationValue
                : default;
            JsonElement plan = root.TryGetProperty("correction_plan", out JsonElement planValue) &&
                               planValue.ValueKind == JsonValueKind.Object
                ? planValue
                : default;
            JsonElement light = plan.ValueKind == JsonValueKind.Object &&
                                plan.TryGetProperty("light", out JsonElement lightValue) &&
                                lightValue.ValueKind == JsonValueKind.Object
                ? lightValue
                : default;
            JsonElement color = plan.ValueKind == JsonValueKind.Object &&
                                plan.TryGetProperty("color", out JsonElement colorValue) &&
                                colorValue.ValueKind == JsonValueKind.Object
                ? colorValue
                : default;
            JsonElement whiteBalance = plan.ValueKind == JsonValueKind.Object &&
                                       plan.TryGetProperty("white_balance", out JsonElement whiteBalanceValue) &&
                                       whiteBalanceValue.ValueKind == JsonValueKind.Object
                ? whiteBalanceValue
                : default;

            _synchronizingManualControls = true;
            ManualExposure = GetNumber(plan, "exposure_ev", GetNestedNumber(recommendations, "exposure_ev", "value", 0.0));
            ManualContrast = GetNumber(light, "contrast", GetNumber(recommendations, "contrast", 0.0));
            ManualHighlights = GetNumber(light, "highlights", GetNumber(recommendations, "highlights", 0.0));
            ManualShadows = GetNumber(light, "shadows", GetNumber(recommendations, "shadows", 0.0));
            ManualTemperature = GetNumber(whiteBalance, "temperature_cast", GetNestedNumber(recommendations, "temperature", "value", 0.0));
            ManualTint = GetNumber(whiteBalance, "tint_cast", GetNestedNumber(recommendations, "tint", "value", 0.0));
            ManualSaturation = GetNumber(color, "saturation", 0.0);
            ManualSharpness = GetNestedNumber(recommendations, "sharpen", "value", 0.0);
        }
        catch (JsonException)
        {
        }
        finally
        {
            _synchronizingManualControls = false;
        }
    }

    private void ApplyMaskedReportSettings(string? settingsJson)
    {
        if (string.IsNullOrWhiteSpace(settingsJson))
            return;

        try
        {
            using JsonDocument document = JsonDocument.Parse(settingsJson);
            if (!document.RootElement.TryGetProperty("mask_report", out JsonElement report) ||
                report.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null)
            {
                return;
            }

            var corrections = new List<JsonElement>();
            CollectObjectsWithProperty(report, "corrections", corrections);
            if (corrections.Count == 0)
                return;

            ManualExposure = AverageCorrection(corrections, "exposure") / 22.0;
            ManualHighlights = -AverageCorrection(corrections, "highlight_reduction") / 0.714;
            ManualShadows = AverageCorrection(corrections, "shadow_lift") / 0.714;
            ManualContrast = AverageCorrection(corrections, "contrast") * 100.0;
        }
        catch (JsonException)
        {
        }
    }

    private static void CollectObjectsWithProperty(
        JsonElement element,
        string propertyName,
        ICollection<JsonElement> results)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            if (element.TryGetProperty(propertyName, out JsonElement property) &&
                property.ValueKind == JsonValueKind.Object)
            {
                results.Add(property);
            }

            foreach (JsonProperty child in element.EnumerateObject())
            {
                CollectObjectsWithProperty(child.Value, propertyName, results);
            }
        }
        else if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement child in element.EnumerateArray())
            {
                CollectObjectsWithProperty(child, propertyName, results);
            }
        }
    }

    private static double AverageCorrection(
        IEnumerable<JsonElement> corrections,
        string propertyName)
    {
        double total = 0.0;
        int count = 0;

        foreach (JsonElement correction in corrections)
        {
            if (correction.TryGetProperty(propertyName, out JsonElement property) &&
                property.ValueKind == JsonValueKind.Number &&
                property.TryGetDouble(out double value))
            {
                total += value;
                count++;
            }
        }

        return count == 0 ? 0.0 : total / count;
    }

    private void ApplyManualSettings(string? settingsJson)
    {
        if (string.IsNullOrWhiteSpace(settingsJson))
            return;

        try
        {
            using JsonDocument document = JsonDocument.Parse(settingsJson);
            JsonElement root = document.RootElement;
            JsonElement adjustments = root.TryGetProperty("adjustments", out JsonElement nested)
                ? nested
                : root;

            ManualExposure = GetNumber(adjustments, "exposure", ManualExposure);
            ManualContrast = GetNumber(adjustments, "contrast", ManualContrast);
            ManualHighlights = GetNumber(adjustments, "highlights", ManualHighlights);
            ManualShadows = GetNumber(adjustments, "shadows", ManualShadows);
            ManualTemperature = GetNumber(adjustments, "temperature", ManualTemperature);
            ManualTint = GetNumber(adjustments, "tint", ManualTint);
            ManualSaturation = GetNumber(adjustments, "saturation", ManualSaturation);
            ManualSharpness = GetNumber(adjustments, "sharpness", ManualSharpness);
        }
        catch (JsonException)
        {
        }
    }

    private static bool TryGetNumber(string? json, string propertyName, out double value)
    {
        value = 0.0;
        if (string.IsNullOrWhiteSpace(json))
            return false;

        try
        {
            using JsonDocument document = JsonDocument.Parse(json);
            if (!document.RootElement.TryGetProperty(propertyName, out JsonElement property) ||
                property.ValueKind != JsonValueKind.Number)
                return false;

            return property.TryGetDouble(out value);
        }
        catch (JsonException)
        {
            return false;
        }
    }

    private static double GetNumber(JsonElement element, string propertyName, double fallback)
    {
        return element.TryGetProperty(propertyName, out JsonElement property) &&
               property.ValueKind == JsonValueKind.Number &&
               property.TryGetDouble(out double value)
            ? value
            : fallback;
    }

    private static double GetNestedNumber(
        JsonElement element,
        string objectName,
        string propertyName,
        double fallback)
    {
        return element.TryGetProperty(objectName, out JsonElement nested) &&
               nested.ValueKind == JsonValueKind.Object
            ? GetNumber(nested, propertyName, fallback)
            : fallback;
    }


    // ================================================================
    // PROCESSING START
    // ================================================================

    private void StartProcessing(
        string message)
    {
        IsBusy = true;

        StatusMessage = message;
    }


    // ================================================================
    // PROCESSING STOP
    // ================================================================

    private void StopProcessing()
    {
        IsBusy = false;


        _processingCts?.Dispose();


        _processingCts = null;

        _activeJobId = null;
    }


    // ================================================================
    // API PROGRESS
    // ================================================================

    private void OnApiProgress(
        ApiJobEvent jobEvent)
    {
        Application.Current.Dispatcher.Invoke(
            () =>
            {
                if (!string.IsNullOrWhiteSpace(
                        jobEvent.Message))
                {
                    StatusMessage =
                        jobEvent.Message;
                }
            });
    }
}