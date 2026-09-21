using System;
using System.ComponentModel;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Threading;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Win32;
using AutoPhotoEditor.ViewModels;
using AutoPhotoEditor.Models;
using AutoPhotoEditor.Theming;
using Wpf.Ui.Appearance;
using Wpf.Ui.Controls;

namespace AutoPhotoEditor
{
    public partial class MainWindow : FluentWindow
    {
        private MainViewModel? ViewModel =>
            DataContext as MainViewModel;

        private bool? _isCompactLayout;


        // =============================================================
        // CONSTRUCTOR
        // =============================================================

        public MainWindow()
        {
            InitializeComponent();

            SystemThemeWatcher.Watch(this);

            DataContext =
                App.Services.GetRequiredService<MainViewModel>();

            ThemeManager.Apply(this, ViewModel?.ThemeMode ?? "System");
            SystemEvents.UserPreferenceChanged += SystemEvents_UserPreferenceChanged;


            if (ViewModel != null)
            {
                ViewModel.PropertyChanged +=
                    ViewModel_PropertyChanged;
            }


            Loaded +=
                MainWindow_Loaded;

            Unloaded +=
                MainWindow_Unloaded;

            SizeChanged +=
                MainWindow_SizeChanged;

            StateChanged +=
                MainWindow_StateChanged;

            UpdateWindowStateVisual();
        }

        private void TitleBar_MouseLeftButtonDown(
            object sender,
            MouseButtonEventArgs e)
        {
            if (e.ChangedButton != MouseButton.Left)
                return;

            if (e.ClickCount == 2)
            {
                ToggleWindowState();
                return;
            }

            if (WindowState == WindowState.Maximized)
                return;

            try
            {
                DragMove();
            }
            catch (InvalidOperationException)
            {
            }
        }

        private void MinimizeButton_Click(
            object sender,
            RoutedEventArgs e)
        {
            WindowState = WindowState.Minimized;
        }

        private void MaximizeButton_Click(
            object sender,
            RoutedEventArgs e)
        {
            ToggleWindowState();
        }

        private void CloseButton_Click(
            object sender,
            RoutedEventArgs e)
        {
            Close();
        }

        private void ToggleWindowState()
        {
            WindowState = WindowState == WindowState.Maximized
                ? WindowState.Normal
                : WindowState.Maximized;
        }

        private void MainWindow_StateChanged(
            object? sender,
            EventArgs e)
        {
            UpdateWindowStateVisual();
        }

        private void UpdateWindowStateVisual()
        {
        }


        // =============================================================
        // VIEWMODEL PROPERTY CHANGED
        // =============================================================

        private void ViewModel_PropertyChanged(
            object? sender,
            PropertyChangedEventArgs e)
        {
            if (e.PropertyName == nameof(MainViewModel.ThemeMode))
            {
                ThemeManager.Apply(this, ViewModel?.ThemeMode ?? "System");
                return;
            }

            if (e.PropertyName != nameof(MainViewModel.OriginalImage) &&
                e.PropertyName != nameof(MainViewModel.EditedImage) &&
                e.PropertyName != nameof(MainViewModel.EditedImagePath) &&
                e.PropertyName != nameof(MainViewModel.ComparisonPosition) &&
                e.PropertyName != nameof(MainViewModel.HistoryIndex) &&
                e.PropertyName != nameof(MainViewModel.HoveredMaskImage))
            {
                return;
            }


            Dispatcher.BeginInvoke(
                DispatcherPriority.Loaded,
                new Action(
                    () =>
                    {
                        UpdateResponsiveLayout();
                        UpdateImageHost();
                        UpdateComparisonVisual();
                    }));
        }


        // =============================================================
        // LOADED
        // =============================================================

        private void MainWindow_Loaded(
            object sender,
            RoutedEventArgs e)
        {
            _ = ViewModel?.StartApiConnectionAsync();

            UpdateResponsiveLayout();
            UpdateImageHost();

            Dispatcher.BeginInvoke(
                DispatcherPriority.Loaded,
                new Action(
                    UpdateComparisonVisual));
        }


        // =============================================================
        // UNLOADED
        // =============================================================

        private async void MainWindow_Unloaded(
            object sender,
            RoutedEventArgs e)
        {
            if (ViewModel != null)
                await ViewModel.StopApiConnectionAsync();

            SystemEvents.UserPreferenceChanged -= SystemEvents_UserPreferenceChanged;
            SystemThemeWatcher.UnWatch(this);

            if (ViewModel != null)
            {
                ViewModel.PropertyChanged -=
                    ViewModel_PropertyChanged;
            }
        }

        private void SystemEvents_UserPreferenceChanged(
            object? sender,
            UserPreferenceChangedEventArgs e)
        {
            if (!string.Equals(
                    ViewModel?.ThemeMode,
                    "System",
                    StringComparison.OrdinalIgnoreCase))
            {
                return;
            }

            Dispatcher.BeginInvoke(
                DispatcherPriority.DataBind,
                new Action(
                    () => ThemeManager.Apply(this, "System")));
        }


        // =============================================================
        // WINDOW RESIZE
        // =============================================================

        private void MainWindow_SizeChanged(
            object sender,
            SizeChangedEventArgs e)
        {
            Dispatcher.BeginInvoke(
                DispatcherPriority.Loaded,
                new Action(
                    () =>
                    {
                        UpdateResponsiveLayout();
                        UpdateImageHost();
                        UpdateComparisonVisual();
                    }));
        }

        private void UpdateResponsiveLayout()
        {
            if (WorkspaceGrid.ActualWidth <= 0)
                return;

            bool compact = WorkspaceGrid.ActualWidth < 1040;
            if (_isCompactLayout == compact)
                return;

            _isCompactLayout = compact;

            WorkspaceGrid.ColumnDefinitions[0].Width =
                compact ? new GridLength(1, GridUnitType.Star) : new GridLength(3, GridUnitType.Star);
            WorkspaceGrid.ColumnDefinitions[1].Width =
                compact ? new GridLength(0) : new GridLength(2, GridUnitType.Star);
            WorkspaceGrid.ColumnDefinitions[1].MinWidth = compact ? 0 : 300;

            WorkspaceGrid.RowDefinitions[0].Height =
                new GridLength(1, GridUnitType.Star);
            WorkspaceGrid.RowDefinitions[1].Height = compact
                ? new GridLength(1, GridUnitType.Star)
                : new GridLength(0);

            Grid.SetColumn(CanvasPane, 0);
            Grid.SetRow(CanvasPane, 0);
            Grid.SetColumn(AdjustmentsPane, compact ? 0 : 1);
            Grid.SetRow(AdjustmentsPane, compact ? 1 : 0);

            TitleContext.Visibility = compact
                ? Visibility.Collapsed
                : Visibility.Visible;

            ApiConnectionStatusTextElement.Visibility = compact
                ? Visibility.Collapsed
                : Visibility.Visible;

            FooterHint.Visibility = compact
                ? Visibility.Collapsed
                : Visibility.Visible;

            CanvasPane.Margin = compact
                ? new Thickness(0)
                : new Thickness(0, 0, 14, 0);
            AdjustmentsPane.Margin = compact
                ? new Thickness(0, 12, 0, 0)
                : new Thickness(0);
        }


        // =============================================================
        // IMAGE HOST SIZE
        // =============================================================

        private void UpdateImageHost()
        {
            if (ViewModel == null)
                return;


            if (ViewModel.OriginalImage == null)
            {
                ImageHost.Width =
                    0;

                ImageHost.Height =
                    0;

                ResetComparisonVisual();

                return;
            }


            // ---------------------------------------------------------
            // IMPORTANT:
            //
            // Host dimensions are ALWAYS based on OriginalImage.
            //
            // Never calculate the host using EditedImage because the
            // preview/API result can have a different pixel resolution.
            // ---------------------------------------------------------

            double imageWidth =
                ViewModel.OriginalImage.PixelWidth;


            double imageHeight =
                ViewModel.OriginalImage.PixelHeight;


            if (imageWidth <= 0 ||
                imageHeight <= 0)
            {
                ImageHost.Width =
                    0;

                ImageHost.Height =
                    0;

                ResetComparisonVisual();

                return;
            }


            double availableWidth =
                PhotoComparisonArea.ActualWidth;


            double availableHeight =
                PhotoComparisonArea.ActualHeight;


            if (availableWidth <= 0 ||
                availableHeight <= 0)
            {
                return;
            }


            availableWidth =
                Math.Max(
                    0,
                    availableWidth - 4);


            availableHeight =
                Math.Max(
                    0,
                    availableHeight - 4);


            if (availableWidth <= 0 ||
                availableHeight <= 0)
            {
                return;
            }


            double imageAspect =
                imageWidth / imageHeight;


            double availableAspect =
                availableWidth / availableHeight;


            double displayWidth;

            double displayHeight;


            if (imageAspect >
                availableAspect)
            {
                displayWidth =
                    availableWidth;


                displayHeight =
                    displayWidth /
                    imageAspect;
            }
            else
            {
                displayHeight =
                    availableHeight;


                displayWidth =
                    displayHeight *
                    imageAspect;
            }


            ImageHost.Width =
                Math.Max(
                    1.0,
                    displayWidth);


            ImageHost.Height =
                Math.Max(
                    1.0,
                    displayHeight);

            MaskHighlightOverlay.Width = ImageHost.Width;
            MaskHighlightOverlay.Height = ImageHost.Height;
            MaskHighlightOverlay.Visibility = ViewModel.HoveredMaskImage == null
                ? Visibility.Collapsed
                : Visibility.Visible;


            // ---------------------------------------------------------
            // Both images MUST occupy the exact same display rectangle.
            // ---------------------------------------------------------

            BeforeImageControl.Width =
                ImageHost.Width;

            BeforeImageControl.Height =
                ImageHost.Height;


            AfterImageControl.Width =
                ImageHost.Width;

            AfterImageControl.Height =
                ImageHost.Height;
        }


        // =============================================================
        // UPDATE COMPARISON VISUAL
        // =============================================================

        private void UpdateComparisonVisual()
        {
            if (ViewModel == null)
                return;


            double width =
                ImageHost.ActualWidth > 0
                    ? ImageHost.ActualWidth
                    : ImageHost.Width;


            double height =
                ImageHost.ActualHeight > 0
                    ? ImageHost.ActualHeight
                    : ImageHost.Height;


            if (double.IsNaN(width) ||
                double.IsNaN(height) ||
                width <= 0 ||
                height <= 0)
            {
                return;
            }


            // ---------------------------------------------------------
            // KEEP BEFORE/AFTER EXACTLY THE SAME SIZE
            // ---------------------------------------------------------

            BeforeImageControl.Width =
                width;

            BeforeImageControl.Height =
                height;


            AfterImageControl.Width =
                width;

            AfterImageControl.Height =
                height;

            MaskHighlightOverlay.Width = width;
            MaskHighlightOverlay.Height = height;


            // ---------------------------------------------------------
            // NO EDITED IMAGE
            //
            // Show only original image.
            // Hide comparison elements.
            // ---------------------------------------------------------

            if (ViewModel.EditedImage == null)
            {
                AfterImageControl.Visibility = Visibility.Collapsed;
                AfterImageClip.Width = 0;
                AfterImageClip.Height = height;
                BeforeImageControl.Opacity = 1.0;
                AfterImageControl.Opacity = 0.0;
                return;
            }


            AfterImageControl.Visibility = Visibility.Visible;

            double position = Math.Clamp(ViewModel.ComparisonPosition, 0.0, 100.0);

            if (position <= 1.0)
            {
                AfterImageClip.Width = 0;
                AfterImageClip.Height = height;
                BeforeImageControl.Opacity = 1.0;
                AfterImageControl.Opacity = 0.0;
                return;
            }

            if (position >= 99.0)
            {
                AfterImageClip.Width = width;
                AfterImageClip.Height = height;
                BeforeImageControl.Opacity = 0.0;
                AfterImageControl.Opacity = 1.0;
                return;
            }

            AfterImageClip.Width = width;
            AfterImageClip.Height = height;
            BeforeImageControl.Opacity = 1.0;
            AfterImageControl.Opacity = 1.0;
        }


        // =============================================================
        // RESET COMPARISON VISUAL
        // =============================================================

        private void ResetComparisonVisual()
        {
            MaskHighlightOverlay.Visibility = Visibility.Collapsed;
            AfterImageClip.Width = 0;
            AfterImageClip.Height = 0;
            BeforeImageControl.Opacity = 1.0;
            AfterImageControl.Opacity = 0.0;
        }

        private void MaskCheckBox_MouseEnter(object sender, MouseEventArgs e)
        {
            ViewModel?.SetHoveredMask((sender as FrameworkElement)?.DataContext as MaskOption);
        }

        private void MaskCheckBox_MouseLeave(object sender, MouseEventArgs e)
        {
            ViewModel?.SetHoveredMask(null);
        }


        // =============================================================
        // RENDER SIZE CHANGED
        // =============================================================

        protected override void OnRenderSizeChanged(
            SizeChangedInfo sizeInfo)
        {
            base.OnRenderSizeChanged(
                sizeInfo);


            Dispatcher.BeginInvoke(
                DispatcherPriority.Loaded,
                new Action(
                    () =>
                    {
                        UpdateImageHost();
                        UpdateComparisonVisual();
                    }));
        }
    }
}