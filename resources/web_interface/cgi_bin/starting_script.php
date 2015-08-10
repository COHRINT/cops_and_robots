
<html>
<body>

<?php
	// $setting = $_POST['setting'];
	$setting = "gazebo";

	if ($setting == "vicon") {
	    $setting_script = '../bash/start_vicon_sierra.sh';
	} elseif ($setting == "gazebo") {
	    $setting_script = '../bash/start_gazebo.sh';
	} else {
	    $setting_script = '../bash/start_python.sh';
	}
	echo('bash '.$setting_script);
	shell_exec('bash '.$setting_script);
?>

</body>
</html>