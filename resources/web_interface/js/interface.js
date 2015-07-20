/**
* Set up primary elements to be manipulated by functions
*/

//Set up primary ROS master
// var rosurl =  'ws://fleming.recuv.org'; //commonly at 128.138.157.84
// var rosMaster = new ROSLIB.Ros({
//   url : rosurl + ':9090'
// });


var rosurl =  'ws://127.0.0.1'; //commonly at 128.138.157.84
var rosMaster = new ROSLIB.Ros({
  url : rosurl + ':9090'
});



//Set up robot objects
var settings = [];
var robotNames = ['deckard','roy','pris','zhora'];
var robotPorts = [':9091',':9092',':9093',':9094'];

var robots = []

for (i = 0;  i < robotNames.length; i++){
	nname = robotNames[i].charAt(0).toUpperCase() + robotNames[i].slice(1);
	var cmdTopic = new ROSLIB.Topic({ ros : rosMaster, name : '/robot_command/' + robotNames[i] , messageType : 'std_msgs/String'});
/* 	var view_topic = new ROSLIB.Topic({ ros : rosMaster, name : '/robot_view/' + robotNames[i] , messageType : 'std_msgs/String'}); */

	robots[i] = {
		url: rosurl + robotPorts[i], 
		active:true, 
		commandTopic:cmdTopic,
/* 		viewTopic:v_topic,		  */
		name:robotNames[i], 
		nicename:nname,
		control:false,
		view:false};
}


/**
* Output to time-stamped interface console
* @param {String} str
*/


/*
* Publishes a message to the input history console
*/
function consoleOut(str){
 	var history = jQuery("#codeHistoryBody");
	var dt = new Date();
	var time = dt.getHours() + ":" + dt.getMinutes() + ":" + dt.getSeconds();

	jQuery("<code>[" + time + "] " + str + "</code> <br />").appendTo("#codeHistoryBody");
	jQuery("#codeHistoryBody").scrollTop(jQuery("#codeHistoryBody")[0].scrollHeight);
}

/*
* Subscribes to question topic and sets up the robot updates panel
*/
function robotUpdateSetup(){

	var questioner = new ROSLIB.Topic({
		ros : rosMaster,
		name: '/questions',
		messageType : 'cops_and_robots_ros/Question'
	});	


	questioner.subscribe(function(message) {
	  updateRobotQuestions(message.questions, message.weights);
	});
}
robotUpdateSetup();
/*
* Updates the robot question panel based on ROS messages
*/
function updateRobotQuestions(publishedQuestions, questionValues){
	
	// Check if we get the right number and type/value of questions
	for(i = 0; i<questionValues.length; i++){
		if((questionValues.length != 5) || (questionValues[i] > 1) || (questionValues[i] < 0)){
			console.log("Not getting properly formed questions.");
			return;
		}
	}

	// Turn them all into precentages
	for(i = 0; i < questionValues.length; i++){
		questionValues[i] = questionValues[i] *100;
	}

	var values = questionValues.slice();	
	var questions = publishedQuestions;

	//Send to pct_weights and robo_questions to index.php to be published in progress bars

	// Set text and weight for questions
	for(i = 0; i < 5; i++){
		$('#question_' + i + ' div p').html(questions[i]);
		$('#question_' + i + ' div').attr("aria-valuenow", values[i].toString());
		$('#question_' + i + ' div').css("width", values[i].toString() + "%");
		console.log('#question_' + i + ' div p')
		console.log(questions[i]);
		console.log(values[i]);
		
	}
	

}



/**
* Select a robot to be controlled via teleoperation
* @param {String} robot
*/
function selectControl(robot){
    for (i = 0;  i < robotNames.length; i++){
    	if (robot === robots[i].nicename){
	    	robots[i].control = true;
    	    settings.commandTopic = robots[i].commandTopic;
    	} else {
	    	robots[i].control = false;
    	}
    }
    
    var str = "Controlling " + robot + "." ;
    consoleOut(str);
}

/**
* Select either Deckard's real camera or the view of any simulated robot
* @param {String} robot
*/
function selectView(robot){
    for (i = 0;  i < robotNames.length; i++){
    	if (robot === robots[i].nicename){
	    	robots[i].view = true;
    	} else{
	    	robots[i].view = false;
    	}
    }
    
    var str = "Viewing " + robot + "." ;
    consoleOut(str);
}


/**
* Get settings parameters from UI and modify model appropriately
*/
function checkSettings(){
	//Select simulation vs. experiment
 	settings.dataSource = jQuery('#settings-source').children('.active').text();

 	var vicon = jQuery("#setting-source-vicon");
	var gazebo= jQuery("#setting-source-gazebo");
	var diffSettings = [vicon, gazebo];
	var vPorts =[":1234", ":1234", ":1234", ":1234", ":1234", ":1234", ":1234"];
	var vTopics = ["/camera/rgb/image_raw", "TOPIC", "TOPIC", "TOPIC", "usb_cam1/image_raw", "usb_cam2/image_raw", "usb_cam3/image_raw"];
	var gPorts =[":1234", ":1234", ":1234", ":1234", ":1234", ":1234", ":1234"];
	var gTopics = ["deckard/camera/image_raw", "pris/camera/image_raw", "roy/camera/image_raw", "zhora/camera/image_raw", "security_camera1/camera/image_raw", "security_camera2/camera/image_raw", "security_camera3/camera/image_raw"]; 
	var ports = [vPorts, gPorts];
	var topics =[vTopics, gTopics];
	var ids = ["#deckard-visual", "#pris-visual", "#roy-visual", "#zhora-visual", "#camera1-visual", "#camera2-visual", "#camera3-visual"];

    for(i = 0; i<diffSettings.length; i++){
    	if(diffSettings[i].parent().hasClass('active') == true){
	      for(j=0; j<ports[i].length; j++){
	      	// $(ids[j]).attr("src", "http://flemming.recov.org"+ports[i][j]+"/stream_viewer?topic="+topics[i][j]);
	      	$(ids[j]).attr("src", "http://127.0.0.1"+ports[i][j]+"/stream_viewer?topic="+topics[i][j]);
	  	  }
	  	  break;	 
		}
    }

	consoleOut('Data source: ' + settings.dataSource );
	
	//Identify active robots
	for (i = 0;  i < robotNames.length; i++){
		var id = '#setting-' + robots[i].name + '-active';
		robots[i].isActive = jQuery(id).hasClass('active');
		
		if (robots[i].name === 'deckard'){
			jQuery('#' + robots[i].name + '-button').addClass('btn-info');
		} else if (robots[i].isActive) {
			jQuery('#' + robots[i].name + '-button').removeClass('btn-default');
			jQuery('#' + robots[i].name + '-button').addClass('btn-primary');
		} else {
			jQuery('#' + robots[i].name + '-button').removeClass('btn-primary');
			jQuery('#' + robots[i].name + '-button').addClass('btn-default');
		}
		if (robots[i].isActive){
			consoleOut(robots[i].nicename + ' is active.'); 
		} else {
			consoleOut(robots[i].nicename + ' is inactive');
		}
	}
}


/**
* Initialize interface parameters (ROS elements, camera, map, etc.)
*/
function init() {
		
	 rosMaster.on('connection', function() {
	    console.log('Connected to websocket server.');
	 });
	
	selectControl('Deckard');
	selectView('Deckard');	
		
	/*---------------------Codebook-------------------*/
	// Establish ROS topic for sending human messages
	var human_sensor = new ROSLIB.Topic({
	    ros : rosMaster,
	    name : '/human_sensor',
	    messageType : 'std_msgs/String'
	});
	
	//Toggle velocity fields
		//Toggle velocity fields

	var posObj = jQuery("#Position_obj");
	var posArea = jQuery("#position_area");
	var moving = jQuery("#Movement");

	var certaintiesObj = document.getElementById("obj_certainties");
	var targetsObj = document.getElementById("obj_targets");
	var positivitiesObj = document.getElementById("obj_positivities");
	var objectRelations = document.getElementById("obj_relations");
	var objects = document.getElementById("obj");
	var certaintiesArea = document.getElementById("area_certainties");
	var targetsArea = document.getElementById("area_targets");
	var positivitiesArea = document.getElementById("area_positivities");
	var areaRelations = document.getElementById("area_relation");
	var areas = document.getElementById("area");
	var certaintiesMv = document.getElementById("mv_certainties");
	var targetsMv = document.getElementById("mv_targets");
	var positivitiesMv = document.getElementById("mv_positivities");
	var movementTypes = document.getElementById("mv_types");
	var movementQualities = document.getElementById("mv_qualities");

	var tabs = [posObj, posArea, moving];
	var certaintyCases = [certaintiesObj, certaintiesArea, certaintiesMv];
	var targetCases = [targetsObj, targetsArea, targetsMv];
	var positivityCases = [positivitiesObj, positivitiesArea, positivitiesMv];
	var discribers = [objectRelations, areaRelations, movementTypes];
	var specifications = [objects, areas, movementQualities];
	
	
	targetsObj.onchange = function(){
		if(targetsObj.value == "nothing"){
			positivitiesObj.options[1].style.display="none";
		}else{
			positivitiesObj.options[1].style.display="inline";
		}
	}

	targetsArea.onchange = function(){
		if(targetsArea.value == "nothing"){
			positivitiesArea.options[1].style.display="none";
		}else{
			positivitiesArea.options[1].style.display="inline";
		}
	}

	targetsMv.onchange = function(){
		if(targetsMv.value == "nothing"){
			positivitiesMv.options[1].style.display="none";
		}else{
			positivitiesMv.options[1].style.display="inline";
		}
	}

	movementTypes.onchange = function(){
		if(movementTypes.value == "stopped"){
			movementQualities.style.display="none";
		}else{
			movementQualities.style.display="inline";
		}
	}

	
	// Publish through ros every time 'submit' is pressed
	jQuery("#human_sensor_button").click(function() { 
	    
	    var s = [];
	    for(i = 0; i<tabs.length; i++){
	    	if(tabs[i].hasClass('active') == true){
	    		s[1] = certaintyCases[i];
			    s[2] = targetCases[i];
			    s[3] = positivityCases[i];
			    s[4] = discribers[i];
			    s[5] = specifications[i];
	    		break;
	    	}
	    }

	    var return_str = [];
	    for (i in s) {
		    if (s[i].style.display == "none"){
		    return_str[i]= "";
		    }
		    else{
		     return_str[i] = s[i].options[s[i].selectedIndex].value;
	 	     if ((i==5))
		 	     return_str[i] = return_str[i] + " ";
		     }
	    }
	    
		var str = 	[""         + return_str[1],
				  	" " 		+ return_str[2],
				  	" " 		+ return_str[3],
				  	" " 		+ return_str[4],
				  	" " 		+ return_str[5],
				  	"."].join('');
	  	var msg = new ROSLIB.Message({data: str});	 
		human_sensor.publish(msg);
		consoleOut(str);
	});
	       	    
	
	/*---------------------Teleoperation-------------------*/
	
	//Command switch
	jQuery('.ctrl-robot').click(function(e) {
	    e.preventDefault();
	    var robot = jQuery(this).parent().prev().text().trim();	    
	    selectControl(robot);
	})
	
	//View Switch
	jQuery('.view-robot').click(function(e) {
	    e.preventDefault();
	    var robot = jQuery(this).parent().prev().text().trim();	    
	    selectView(robot);
	})	


	 // Initialize the teleop.
    var teleop = new KEYBOARDTELEOP.Teleop({
      ros : rosMaster,
      topic : '/cmd_vel'
    });
	
/*
	// Listen to key presses and publish through ROS 
	document.addEventListener('keypress', function(event) {
	    char = String.fromCharCode(event.which);	    	    
	    var cmd = new ROSLIB.Message({data: char});	 	    
	    settings.commandTopic.publish(cmd);
	}, true);
	
*/
/*
	// Create subscriber to /battery topic
	var deckard_batteryListener = new ROSLIB.Topic({
	ros : deckard_ros,
	name : '/battery',
	messageType : 'cops_and_robots/battery'
	});
	
	var roy_batteryListener = new ROSLIB.Topic({
	ros : roy_ros,
	name : '/battery',
	messageType : 'cops_and_robots/battery'
	});
*/
	
	// Update battery charge values
	/*
	deckard_batteryListener.subscribe(function(message) {
	pct = (message.charge / message.capacity).toFixed(2) * 100;
	jQuery('#deckardBattery').html(pct + '%');
	});
	
	roy_batteryListener.subscribe(function(message) {
	pct = (message.charge / message.capacity).toFixed(2) * 100;
	jQuery('#royBattery').html(pct + '%');
	});
	*/
	
	
/*	
	// Update battery charge values
	poseListener.subscribe(function(message) {
	jQuery('#2Dmap').html(pct + '%');
	console.log('Received message on ' + poseListener.name + ': ' + message.translation);
	});
*/	
}

/**
* Uses ajax to call bash script
*/
function callScript(arg){
    jQuery.ajax({ 
    	type: "POST",
        url: "/bash/start_stop.sh",
        dataType: "script",
        data:{setting: arg},  
        async: true,
        success: function(body){  
            alert('response received: ' + arg);              
        } 
    }); 

}

/**
* Load js elements once document is ready
*/
jQuery(document).ready(function(){
	jQuery("[data-toggle=tooltip]").tooltip({
	    selector: '[rel=tooltip]',
	    container:'body'
	});
		
	checkSettings();
	init();
});

jQuery('#settings-save').click(function(){
	checkSettings();
	$('#settings').modal('hide');
});

jQuery('#start-stop').click(function(){

	var vicon = jQuery("#setting-source-vicon");
	var gazebo= jQuery("#setting-source-gazebo");
	var diffSettings = [vicon, gazebo];
	var activeSet = ['vicon', 'gazebo'];

    jQuery(this).toggleClass("btn-success");
    jQuery(this).toggleClass("btn-danger");			        
    jQuery(this).children("span").toggleClass("glyphicon-play");
    jQuery(this).children("span").toggleClass("glyphicon-stop");	

    var setting = '';
    for(i = 0; i<diffSettings.length; i++){
    	if(diffSettings[i].parent().hasClass('active') == true){
    		setting = activeSet[i];
    		break;
    	}
    }

    callScript(setting);

	// var childExec = require('child_process');
	// console.log(childExec);
	// console.log(childExec.execFile);
	// function getMethods(obj) {
	//   var result = [];
	//   for (var id in obj) {
	//     try {
	//       if (typeof(obj[id]) == "function") {
	//         result.push(id + ": " + obj[id].toString());
	//       }
	//     } catch (err) {
	//       result.push(id + ": inaccessible");
	//     }
	//   }
	//   return result;
	// }
	// console.log(getMethods(childExec));

	// childExec.execFile('./bash/start_stop.sh', [setting], {}, function (error, stdout, stderr) {
	// 	console.log('Hi!!')
	// });
         
});