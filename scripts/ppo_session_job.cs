// Windows PowerShell 5.1: one owned Job Object, suspended assignment, wall watchdog.
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

public sealed class PpoSessionJob : IDisposable {
    [StructLayout(LayoutKind.Sequential)] struct BasicLimits {
        public long ProcessTime, JobTime; public uint Flags;
        public UIntPtr Minimum, Maximum; public uint Processes; public UIntPtr Affinity;
        public uint Priority, Scheduling;
    }
    [StructLayout(LayoutKind.Sequential)] struct IoCounters { public ulong A,B,C,D,E,F; }
    [StructLayout(LayoutKind.Sequential)] struct ExtendedLimits {
        public BasicLimits Basic; public IoCounters Io; public UIntPtr ProcessMemory, JobMemory, PeakProcess, PeakJob;
    }
    [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)] struct Startup {
        public int Size; public string Reserved, Desktop, Title;
        public uint X,Y,Width,Height,XCount,YCount,Fill,Flags; public short Show,Reserved2;
        public IntPtr ReservedPointer, Input, Output, Error;
    }
    [StructLayout(LayoutKind.Sequential)] struct ProcessInfo { public IntPtr Process, Thread; public uint Id, ThreadId; }
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern IntPtr CreateJobObject(IntPtr attributes, string name);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool SetInformationJobObject(IntPtr job, int type, ref ExtendedLimits info, uint size);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool TerminateJobObject(IntPtr job, uint code);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool TerminateProcess(IntPtr process, uint code);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool GetExitCodeProcess(IntPtr process, out uint code);
    [DllImport("kernel32.dll", SetLastError=true)] static extern uint WaitForSingleObject(IntPtr handle, uint milliseconds);
    [DllImport("kernel32.dll", SetLastError=true)] static extern uint ResumeThread(IntPtr thread);
    [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern bool CreateProcess(
        string app, StringBuilder command, IntPtr processAttributes, IntPtr threadAttributes, bool inherit,
        uint flags, IntPtr environment, string cwd, ref Startup startup, out ProcessInfo process);
    readonly object gate = new object();
    IntPtr job, process; Timer timer; volatile bool forced;
    public bool Forced { get { return forced; } }
    public int ProcessId { get; private set; }
    public static string Quote(string value) {
        var s = new StringBuilder("\""); int slashes=0;
        foreach (char c in value) {
            if (c=='\\') { slashes++; continue; }
            if (c=='\"') s.Append('\\', slashes*2+1).Append(c);
            else s.Append('\\', slashes).Append(c);
            slashes=0;
        }
        return s.Append('\\',slashes*2).Append('"').ToString();
    }
    public static double Remaining(long start, double cap) {
        return Math.Max(0, cap-(Stopwatch.GetTimestamp()-start)/(double)Stopwatch.Frequency);
    }
    public PpoSessionJob(string command, string cwd, long sessionStart, double limitSeconds) {
        try {
            double remaining=Remaining(sessionStart, limitSeconds);
            if (remaining<=0) throw new TimeoutException("Session budget exhausted before process launch.");
            job=CreateJobObject(IntPtr.Zero,null);
            if (job==IntPtr.Zero) throw new Win32Exception();
            var limits=new ExtendedLimits(); limits.Basic.Flags=0x2000; // KILL_ON_JOB_CLOSE
            if (!SetInformationJobObject(job,9,ref limits,(uint)Marshal.SizeOf(limits))) throw new Win32Exception();
            timer=new Timer(delegate { lock(gate) { if(job!=IntPtr.Zero) { forced=true; TerminateJobObject(job,124); } } },
                null, TimeSpan.FromSeconds(remaining), Timeout.InfiniteTimeSpan);
            var startup=new Startup(); startup.Size=Marshal.SizeOf(startup);
            ProcessInfo info;
            if (!CreateProcess(null,new StringBuilder(command),IntPtr.Zero,IntPtr.Zero,false,0x08000004,
                IntPtr.Zero,cwd,ref startup,out info)) throw new Win32Exception(); // NO_WINDOW | SUSPENDED
            process=info.Process; ProcessId=(int)info.Id;
            try {
                if (!AssignProcessToJobObject(job,process)) throw new Win32Exception();
                if (Remaining(sessionStart,limitSeconds)<=0 || forced) throw new TimeoutException("Deadline before resume.");
                if (ResumeThread(info.Thread)==uint.MaxValue) throw new Win32Exception();
            } catch { TerminateProcess(process,124); throw; }
            finally { CloseHandle(info.Thread); }
        } catch { Dispose(); throw; }
    }
    public bool Wait(int milliseconds) { return WaitForSingleObject(process,(uint)milliseconds)==0; }
    public int ExitCode { get { uint code; if(!GetExitCodeProcess(process,out code)) throw new Win32Exception(); return (int)code; } }
    public void Dispose() {
        lock(gate) {
            if(timer!=null) { timer.Dispose(); timer=null; }
            if(job!=IntPtr.Zero) { CloseHandle(job); job=IntPtr.Zero; }
            if(process!=IntPtr.Zero) { CloseHandle(process); process=IntPtr.Zero; }
        }
    }
}
