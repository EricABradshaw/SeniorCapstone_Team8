# Initialization
[XML]$config = Get-Content .\config.xml
$nodeBase = $config.SelectSingleNode("//Setting[@Name='nodeurl']").Value
$goodB64_1 = Get-Content .\goodb64_1.txt
$goodB64_2 = Get-Content .\goodb64_2.txt
$bigB64 = Get-Content .\bigb64.txt
$smallB64 = Get-Content .\smallb64.txt
$Script:output = @()

function test {
  param(
    $caseName,
    $inputBody,
    $expectedResult
  )

  try {
    $res = Invoke-WebRequest "$($nodeBase)/api/hide" -Method 'Post' -Body $inputBody -ContentType "application/json" -ErrorAction Stop -Verbose

    $status = $res.StatusCode
    $serverMessage = ""
    if ($null -eq $status) {
      $status = "no response"
    }
    elseif ($status -eq 400) {
      $serverMessage = $res.data.error
    }

    $result = "fail"
    if ($res.StatusCode -eq $expectedResult) {
      $result = "pass"
    }

    $responseContent = $res.Content | ConvertFrom-Json

    $stegoImageExists = "no"
    if ($responseContent -is [System.Management.Automation.PSCustomObject]) {
      if ($responseContent.PSObject.Properties.Name -contains "stegoImage") {
        $stegoImageExists = "yes"
      }
    }

    $Script:output += [PSCustomObject]@{
      TestCase        = $caseName
      Expected_Result = $expectedResult
      Actual_Result   = $status
      Recieved_Stego  = $stegoImageExists
      Test_Result     = $result
      Server_Message  = $serverMessage
    }
  }
  catch {
    $result = "fail"
    if($expectedResult -eq 400){
      $result = "pass"
    }
    $errorMessage = $_.Exception.Message
    $Script:output += [PSCustomObject]@{
      TestCase        = $caseName
      Expected_Result = $expectedResult
      Actual_Result   = 400
      Recieved_Stego  = "no"
      Test_Result     = $result
      Server_Message  = $errorMessage
    }
  }
}



# Case - Good Request
$body = @{
  coverImageData  = $goodB64_1
  secretImageData = $goodB64_2
  sliderValue     = 90
} | ConvertTo-Json
test "Good Request" $body 200

# Case - Empty Cover
$body = @{
  coverImageData  = ""
  secretImageData = $goodB64_2
  sliderValue     = 90
} | ConvertTo-Json
test "Empty Cover" $body 400

# Case - Empty Secret
$body = @{
  coverImageData  = $goodB64_1
  secretImageData = ""
  sliderValue     = 90
} | ConvertTo-Json
test "Empty Secret" $body 400

#Case - Empty Secret and cover
$body = @{
  coverImageData  = ""
  secretImageData = ""
  sliderValue     = 90
} | ConvertTo-Json
test "Empty Secret and Cover" $body 400

#Case - Invalid Slider
$body = @{
  coverImageData  = $goodB64_1
  secretImageData = $goodB64_2
  sliderValue     = -1
} | ConvertTo-Json
test "Invalid Slider" $body 400

# Case - Missing Cover
$body = @{
  secretImageData = $goodB64_2
  sliderValue     = 90
} | ConvertTo-Json
test "Missing Cover" $body 400

# Case - Missing Secret
$body = @{
  coverImageData = $goodB64_1
  sliderValue    = 90
} | ConvertTo-Json
test "Missing Secret" $body 400

#Case - Missing Secret and cover
$body = @{
  sliderValue = 90
} | ConvertTo-Json
test "Missing Secret and Cover" $body 400

# Case - Missing Slider
$body = @{
  coverImageData  = $goodB64_1
  secretImageData = $goodB64_2
} | ConvertTo-Json
test "Missing Slider" $body 200

# Case - Large Cover
$body = @{
  coverImageData  = $bigB64
  secretImageData = $goodB64_2
  sliderValue     = 90
} | ConvertTo-Json
test "Large Cover" $body 200

# Case - Large Secret
$body = @{
  coverImageData  = $goodB64_1
  secretImageData = $bigB64
  sliderValue     = 90
} | ConvertTo-Json
test "Large Secret" $body 400

# Case - Large Secret and Cover
$body = @{
  coverImageData  = $bigB64
  secretImageData = $bigB64
  sliderValue     = 90
} | ConvertTo-Json
test "Large Secret and Cover" $body 400

# Case - Small Secret
$body = @{
  coverImageData  = $goodB64_1
  secretImageData = $smallB64
  sliderValue     = 90
} | ConvertTo-Json
test "Small Secret" $body 400

# Case - Small Cover
$body = @{
  coverImageData  = $smallB64
  secretImageData = $goodB64_2
  sliderValue     = 90
} | ConvertTo-Json
test "Small Cover" $body 400

# Case - Small Secret and Cover
$body = @{
  coverImageData  = $smallB64
  secretImageData = $smallB64
  sliderValue     = 90
} | ConvertTo-Json
test "Small Secret and Cover" $body 400

#Case - Empty Request
$body = @{} | ConvertTo-Json
test "Empty Request" $body 400

# Write output to console and export as csv
Write-Output $output
$output | Export-Csv .\HideResults.csv -NoTypeInformation