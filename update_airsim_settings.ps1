<#
.SYNOPSIS
    This script simplifies modifying AirSim global settings. Fundamentally, it
    copies a given file into a `settings.json` file where AirSim expects it.
.DESCRIPTION
    .
.PARAMETER File
    The file to copy to AirSim Settings
.PARAMETER OutDir
    The directory to copy the given file to, under the name "settings.json".
    By default, this is the directory AirSim looks for settigns globally.
    If using a compiled version of the simulation, you may want to change this
    to the directory of the executable, since AirSim looks there first.
.PARAMETER NumDrones
    This is part of a ease-of-use functionality that automatically creates N
    drones compatible with what our simulation Docker container expects. The
    settings of each drone is copied from the drone settings provided by the given
    File, but the properties "ControlPort", "UdpPort", "X", and "Y" are adjusted
    automatically and may be overridden.

    This parameter expects a grid size input, or ROW,COL.
.PARAMETER XSep
    Given that NumDrones is provided, determines the x-separation between drones
    in meters. This parameters is 3 by default.
.PARAMETER YSep
    Given that NumDrones is provided, determines the y-separation between drones
    in meters. This parameters is 3 by default.
.PARAMETER StartControlPort
    Given that NumDrones is provided, determines the control port of the first drone.
    Each successive drone has a port ten higher than the previous. It is 9002 by default.

    e.g., drone ports will be {9002, 9012, 9022, 9032, ...} if this value is 
    9002
.PARAMETER StartUdpPort
    Given that NumDrones is provided, determines the UDP port of the first drone.
    Each successive drone has a port ten higher than the previous. It is 9003 by default.

    e.g., drone ports will be {9003, 9013, 9023, 9033, ...} if this value is 
    9003
.EXAMPLE
    C:\SUAS-2025> ./update_airsim_settings.ps1 ./my_settings.json
    Simplest case copying the contents of ./my_settings.json to the default AirSim settigns.json.

    C:\SUAS-2025> ./update_airsim_settings.ps1 ./my_settings.json -NumDrones 5,5
    Example of automatically creating a 5 by 5 grid of drones. The settigns in my_settings.json 
    will be used as a template.
#>

# This script copies the content of the provided file into AirSim's settings
# Simulation only works on Windows, hence a powershell script

param(
    [Parameter(Mandatory, Position=0)]
    [string]$File,  # this is positional (e.g., ./update_airsim_settings.ps1 file.json)
    
    [string]$OutDir = [System.IO.Path]::Combine([Environment]::GetFolderPath('Personal'), "AirSim"),

    [Parameter(ParameterSetName="ndrones")]
    [int[]]$NumDrones,
    [Parameter(ParameterSetName="ndrones")]
    [int]$XSep = 3,
    [Parameter(ParameterSetName="ndrones")]
    [int]$YSep = 3,
    [Parameter(ParameterSetName="ndrones")]
    [int]$StartControlPort = 9002,
    [Parameter(ParameterSetName="ndrones")]
    [int]$StartUdpPort = 9003

)

# returns the value of an object's property or a specifed default value if the property is null
function Get-ObjectPropertyOrDefault {
    param(
        [Parameter(Mandatory=$true)]
        $Object,

        [Parameter(Mandatory=$true)]
        [string]$PropertyName,

        [Parameter(Mandatory=$true)]
        $DefaultValue
    )

    # Check if the property actually exists on the object
    if ($Object.$PropertyName) {
        # Return the actual value of the property
        return $Object.$PropertyName
    }
    else {
        # Return the default value
        return $DefaultValue
    }
}

$settings = Get-Content "$File"

if ($null -ne $NumDrones) {

    # validate NumDrones
    if ($NumDrones.Count -ne 2) {
        Write-Error -Message "NumDrones must be an array of size 2! NumDrones should be given in NumRows,NumCols format." -Category InvalidArgument
        exit
    }
    if ($NumDrones[0] -lt 1 -or $NumDrones[1] -lt 1) {
        Write-Error -Message "Both values in NumDrones must be integers greater than 0!" -Category InvalidArgument
        exit

    }

    # create drones
    $settingsJSON = $settings | ConvertFrom-Json  # convert the settings to a PowerShell object

    # copy the first vehicle in Vehicles to $templateDrone
    $templateDrone = @{}
    $($settingsJSON.Vehicles.PsObject.Properties.Value | Select-Object -First 1).PsObject.Properties | ForEach-Object { $templateDrone[$_.Name] = $_.Value }

    # find base X and Y offsets
    $baseX = Get-ObjectPropertyOrDefault $templateDrone X 0
    $baseY = Get-ObjectPropertyOrDefault $templateDrone Y 0

    $settingsJSON.Vehicles = @{}

    # create drone grid
    for ($row = 0; $row -lt $NumDrones[0]; $row++) {
        for ($col = 0; $col -lt $NumDrones[1]; $col++) {
            $droneId = $NumDrones[1] * $row + $col
            $newDrone = $templateDrone.Clone()

            $newDrone["X"] = $baseX + ($XSep * $row)
            $newDrone["Y"] = $baseY + ($YSep * $col)
            $newDrone["UdpPort"] = $StartUdpPort + (10 * $droneId)
            $newDrone["ControlPort"] = $StartControlPort + (10 * $droneId)

            $settingsJSON.Vehicles["Copter$($droneId + 1)"] = $newDrone

        }
    }
    $settings = $settingsJSON | ConvertTo-Json -Depth 100

}

Write-Output "$settings" > "$outdir\settings.json"

Write-Output "Updated $outdir\settings.json"

Write-Output "Settings updated!"
