# Initialization
[XML]$config = Get-Content .\config.xml
$nodeBase = $config.SelectSingleNode("//Setting[@Name='nodeurl']").Value
$goodB64 = Get-Content .\goodb64_1.txt
$bigB64 = Get-Content .\bigb64.txt
$smallB64 = Get-Content .\smallb64.txt
$Script:output = @()

function test{
  param(
    $caseName,
    $inputBody,
    $expectedResult
  )

  $res = Invoke-WebRequest "$($nodeBase)/api/recommendation" -Method 'Post' -Body $body -ContentType "application/json" -Verbose

  $status = $res.StatusCode
  $serverMessage = ""
  if($null -eq $status){
    $status = "no response"
  }
  elseif ($status -eq 400) {
    $serverMessage = $res.data.error
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
    Server_Message = $serverMessage
  }

}


# Case - Good Request
$body = @{
  secretImage = $goodB64
  sliderValue = 75
} | ConvertTo-Json
test "Good Request" $body 200

# Case - Invalid Slider
$body = @{
  secretImage = $goodB64
  sliderValue = -1
} | ConvertTo-Json
test "Invalid Slider" $body 400

# Case - Missing Slider
$body = @{
  secretImage = $goodB64
} | ConvertTo-Json
test "Missing Slider" $body 200

# Case - Small Secret
$body = @{
  secretImage = $smallB64
  sliderValue = 75
} | ConvertTo-Json
test "Small Secret" $body 200

# Case - Large Secret
$body = @{
  secretImage = $bigB64
  sliderValue = 75
} | ConvertTo-Json
test "Large Secret" $body 200

# Case - Missing Secret
$body = @{
  sliderValue = 75
} | ConvertTo-Json
test "Missing Secret" $body 400

# Case - Empty Request
$body = @{} | ConvertTo-Json
test "Empty Request" $body 400

# Write output to console and export as csv
Write-Output $output
$output | Export-Csv .\RecommendResults.csv -NoTypeInformation