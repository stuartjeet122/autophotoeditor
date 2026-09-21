using System.Windows;
using AutoPhotoEditor.Api;
using AutoPhotoEditor.ViewModels;
using Microsoft.Extensions.DependencyInjection;

namespace AutoPhotoEditor
{
    public partial class App : Application
    {
        private readonly ServiceProvider _services;

        public static IServiceProvider Services { get; private set; } = null!;


        public App()
        {
            var services = new ServiceCollection();

            services.AddSingleton<AutoPhotoEditorApiConfig>();
            services.AddSingleton<ApiHttpHelper>();
            services.AddSingleton<AutoPhotoEditorApiClient>();
            services.AddSingleton<ApiWebSocketConnection>();
            services.AddTransient<MainViewModel>();
            services.AddTransient<MainWindow>();

            _services = services.BuildServiceProvider();
            Services = _services;
        }

        protected override void OnStartup(StartupEventArgs e)
        {
            base.OnStartup(e);

            try
            {
                var mainWindow =
                    Services.GetRequiredService<MainWindow>();

                MainWindow = mainWindow;
                mainWindow.Show();
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    $"AutoPhotoEditor could not start.\n\n" +
                    $"{ex.Message}",
                    "AutoPhotoEditor",
                    MessageBoxButton.OK,
                    MessageBoxImage.Error
                );

                Shutdown(-1);
            }
        }

        protected override async void OnExit(ExitEventArgs e)
        {
            ApiWebSocketConnection? connection =
                Services.GetService<ApiWebSocketConnection>();

            if (connection != null)
                await connection.DisposeAsync();

            _services.Dispose();

            base.OnExit(e);
        }
    }
}