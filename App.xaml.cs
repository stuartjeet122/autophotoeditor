using System.Windows;
using AutoPhotoEditor.Api;
using AutoPhotoEditor.ViewModels;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace AutoPhotoEditor
{
    public partial class App : Application
    {
        private readonly IHost _host;

        public static IServiceProvider Services { get; private set; } = null!;


        public App()
        {
            _host = Host.CreateDefaultBuilder()
                .ConfigureServices((context, services) =>
                {

                    services.AddSingleton<AutoPhotoEditorApiConfig>();

                    services.AddSingleton<ApiHttpHelper>();

                    services.AddSingleton<AutoPhotoEditorApiClient>();

                    // ViewModels
                    services.AddTransient<MainViewModel>();

                    // Views
                    services.AddTransient<MainWindow>();
                })
                .Build();

            Services = _host.Services;
        }

        protected override async void OnStartup(StartupEventArgs e)
        {
            base.OnStartup(e);

            try
            {
                await _host.StartAsync();

                var mainWindow =
                    Services.GetRequiredService<MainWindow>();

                mainWindow.Show();
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    $"AutoPhotoEditor API failed to start.\n\n" +
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
            await _host.StopAsync();

            _host.Dispose();

            base.OnExit(e);
        }
    }
}