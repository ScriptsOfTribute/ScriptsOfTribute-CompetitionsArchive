# PatronZero

Martin Niederl, Johannes Kepler University Linz

The bot entry point is `PatronZeroSubmission.PatronZeroBot`. It has a parameterless constructor and targets .NET 8.

## Build

The project needs the official ScriptsOfTribute Core source. From this directory, run:

```powershell
dotnet build PatronZeroBot.csproj -c Release -p:ScriptsOfTributeCore=C:\path\to\ScriptsOfTribute-Core
```

NuGet restores TorchSharp and its runtime libraries during the build.

## Install in GameRunner

1. Copy `PatronZeroBot.dll` and the `model` directory from the build output into GameRunner's `Bots` directory.
2. Copy the dependency DLLs from the build output beside GameRunner, but keep GameRunner's own `TalesOfTribute.dll`.
3. Copy the build output's `runtimes` directory beside GameRunner.

GameRunner can load the bot as `PatronZeroBot` or `PatronZeroSubmission.PatronZeroBot`.
