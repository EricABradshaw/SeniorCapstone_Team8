# Initialization
[XML]$config = Get-Content .\config.xml
$nodeBase = $config.SelectSingleNode("//Setting[@Name='nodeurl']").Value
$goodB64_1 = Get-Content .\goodb64_1.txt
$goodB64_2 = Get-Content .\goodb64_2.txt
$bigB64 = Get-Content .\bigb64.txt
$smallB64 = Get-Content .\smallb64.txt
$Script:output = @()

function test{
  param(
    $caseName,
    $inputBody,
    $expectedResult
  )

  $res = Invoke-WebRequest "$($nodeBase)/api/hide" -Method 'Post' -Body $body -ContentType "application/json" -Verbose

  $status = $res.StatusCode
  if($null -eq $status){
    $status = "no response"
  }

  $result = "fail"
  if($res.StatusCode -eq $expectedResult){
    $result = "pass"
  }

  $responseContent = $res.Content | ConvertFrom-Json

  $stegoImageExists = "no"
  if ($responseContent -is [System.Management.Automation.PSCustomObject]) {
    if($responseContent.PSObject.Properties.Name -contains "stegoImage"){
      $stegoImageExists = "yes"
    }
  }

  $Script:output += [PSCustomObject]@{
    TestCase = $caseName
    Expected_Result = $expectedResult
    Actual_Result = $status
    Recieved_Stego = $stegoImageExists
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

#Case 4 - Invalid Slider
$body = @{
  coverImageData = $goodB64_1
  secretImageData = $goodB64_2
  sliderValue = -1
} | ConvertTo-Json
test "Invalid Slider" $body 400

# Case 5 - Missing Cover
$body = @{
  secretImageData = $goodB64_2
  sliderValue = 90
} | ConvertTo-Json
test "Missing Cover" $body 400

# Case 6 - Missing Secret
$body = @{
  coverImageData = $goodB64_1
  sliderValue = 90
} | ConvertTo-Json
test "Missing Secret" $body 400

# Case 7 - Missing Slider
$body = @{
  coverImageData = $goodB64_1
  secretImageData = $goodB64_2
} | ConvertTo-Json
test "Missing Slider" $body 400

# Case 8 - Large Cover
$body = @{
  coverImageData = $bigB64
  secretImageData = $goodB64_2
  sliderValue = 90
} | ConvertTo-Json
test "Large Cover" $body 200

# Case 9 - Large Secret
$body = @{
  coverImageData = $goodB64_1
  secretImageData = $bigB64
  sliderValue = 90
} | ConvertTo-Json
test "Large Secret" $body 400

# Case 10 - Large Secret and Cover
$body = @{
  coverImageData = $bigB64
  secretImageData = $bigB64
  sliderValue = 90
} | ConvertTo-Json
test "Large Secret and Cover" $body 400

# Case 11 - Small Secret
$body = @{
  coverImageData = $goodB64_1
  secretImageData = $smallB64
  sliderValue = 90
} | ConvertTo-Json
test "Small Secret" $body 400

# Case 12 - Small Cover
$body = @{
  coverImageData = $smallB64
  secretImageData = $goodB64_2
  sliderValue = 90
} | ConvertTo-Json
test "Small Cover" $body 400

# Case 13 - Small Secret and Cover
$body = @{
  coverImageData = $smallB64
  secretImageData = $smallB64
  sliderValue = 90
} | ConvertTo-Json
test "Small Secret and Cover" $body 400

# Write output to console and export as csv
Write-Output $output
$output | Export-Csv .\HideResults.csv -NoTypeInformation