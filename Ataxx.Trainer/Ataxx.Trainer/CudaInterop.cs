using System;
using System.Diagnostics;

public static class CudaInterop
{
    public static void PrintGpuMemory()
    {
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = "nvidia-smi",
                Arguments = "--query-gpu=memory.used,memory.free --format=csv,noheader,nounits",
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var process = Process.Start(psi);
            if (process == null)
            {
                Console.WriteLine("Could not start nvidia-smi process.");
                return;
            }
            string output = process.StandardOutput.ReadToEnd();
            process.WaitForExit();

            Console.WriteLine($"GPU Memory (Used, Free): {output.Trim()} MB");
        }
        catch (System.ComponentModel.Win32Exception)
        {
            Console.WriteLine("nvidia-smi.exe not found. Make sure it is in your system's PATH.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred while running nvidia-smi: {ex.Message}");
        }
    }
}