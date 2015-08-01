<html>
<body>

<?php

	$setting = $_POST['setting'];

	if ($setting == "vicon") {
	    $setting_script = '../bash/start_vicon_sierra.sh';
	} elseif ($setting == "gazebo") {
	    $setting_script = '../bash/start_gazebo.sh';
	} else {
	    $setting_script = '../bash/start_python.sh';
	}
	echo('bash '.$setting_script);
	// exec('bash '.$setting_script);
	exec('bash http:127.0.0.1/optimized_web_interface/bash/start_gazebo.sh');
?>

</body>
</html>