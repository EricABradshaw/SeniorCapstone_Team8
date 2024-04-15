[XML]$config = Get-Content .\config.xml
$nodeBase = $config.SelectSingleNode("//Setting[@Name='nodeurl']").Value
$num = [int]$config.SelectSingleNode("//Setting[@Name='stressNum']").Value
$goodB64_1 = Get-Content .\goodb64_1.txt
$goodB64_2 = Get-Content .\goodb64_2.txt
$Global:output = @()

function Test-Stress {

  param(
      # Script block to be ran by test
      [ScriptBlock]
      $ScriptBlock,

      # Determines if the test should be ran in the background
      [Switch]
      $AsJob,

      # Number of times to run the test. Default is 100
      [uint32]
      $Stress = 100
  )

  process {
      1..$Stress | ForEach-Object -Parallel {
        Write-Output "Request " + $_
          $start = [DateTime]::Now
          $end = [DateTime]::Now

          [PSCustomObject]@{
              Script = $ScriptBlock
              Results = $results
              Start = $start
              End = $end
              ExecutionTime = $end - $start
          }
      } -ThrottleLimit $Stress -AsJob:$AsJob
  }
}


$Global:body = @{
  secretImageData = $goodB64_1
  coverImageData = $goodB64_2
  sliderValue = 75
} | ConvertTo-Json

Test-Stress -Stress $num -ScriptBlock{
  Invoke-WebRequest "$($nodeBase)/api/hide" -Method 'Post' -Body $Global:body -ContentType "application/json" -Verbose} | 
  ForEach-Object {$_.ExecutionTime.TotalMilliseconds} |
  Measure-Object -Average -Minimum -Maximum -StandardDeviation >> StressResults.txt