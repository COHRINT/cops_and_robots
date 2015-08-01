<html>
<body>

<?php

	$setting = $_POST['setting'];

	if ($setting == "vicon") {
	    $setting_script = '../bash/stop_vicon.sh';
	} elseif ($setting == "gazebo") {
	    $setting_script = '../bash/stop_gazebo.sh';
	} else {
	    $setting_script = '../bash/stop_python.sh';
	}
	echo('bash '.$setting_script);
	exec('bash '.$setting_script);
?>

</body>
</html>