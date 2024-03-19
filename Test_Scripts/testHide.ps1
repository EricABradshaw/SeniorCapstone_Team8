# Initialization
[XML]$config = Get-Content .\config.xml
$nodeBase = $config.SelectSingleNode("//Setting[@Name='nodeurl']").Value
$goodB64_1 = Get-Content .\goodb64_1.txt
$goodB64_2 = Get-Content .\goodb64_2.txt
$Script:output = @()

function test{
  param(
    $caseName,
    $inputBody,
    $expectedResult
  )

  $res = Invoke-WebRequest "$($nodeBase)/api/hide" -Method 'Post' -Body $body -ContentType "application/json" -Verbose

  $result = "fail"
  if($res.StatusCode -eq $expectedResult){
    $result = "pass"
  }

  $Script:output += [PSCustomObject]@{
    TestCase = $caseName
    Expected_Result = $expectedResult
    Actual_Result = $res.StatusCode
    Test_Result = $result
  }

}


# Case 1 - Good Request
$body = @{
  coverImageData = $goodB64_1
  secretImageData = $goodB64_2
  sliderValue = 90
} | ConvertTo-Json
test "Good Request" $body 200

# Case 2 - Empty Cover
$body = @{
  coverImageData = ""
  secretImageData = $goodB64_2
  sliderValue = 90
} | ConvertTo-Json
test "Empty Cover" $body 400

# Case 3 - Empty Secret
$body = @{
  coverImageData = $goodB64_1
  secretImageData = ""
  sliderValue = 90
} | ConvertTo-Json
test "Empty Secret" $body 400

# Write output to console and export as csv
Write-Output $output
$output | Export-Csv .\HideResults.csv -NoTypeInformation