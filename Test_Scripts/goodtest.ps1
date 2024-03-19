[XML]$config = Get-Content .\config.xml
$nodeBase = $config.SelectSingleNode("//Setting[@Name='nodeurl']").Value
$goodRequest = Get-Content .\goodHideRequest.json
$goodB64_1 = Get-Content .\goodb64_1.txt
$goodB64_2 = Get-Content .\goodb64_2.txt
$Body = @{
  coverImageData = $goodB64_1
  secretImageData = $goodB64_2
  sliderValue = 90
} | ConvertTo-Json

$goodReqBody = @{
  coverImageData = $goodB64_1
  secretImageData = $goodB64_2
  sliderValue = 90
} | ConvertTo-Json

$res = Invoke-WebRequest "$($nodeBase)/api/hide" -Method 'Post' -Body $goodReqBody -ContentType "application/json" -Verbose
$res.StatusCode
