/**
* Set up primary elements to be manipulated by functions
*/

//Set up primary ROS master
var rosurl =  'ws://fleming.recuv.org'; //commonly at 128.138.157.84
var rosMaster = new ROSLIB.Ros({
  url : rosurl + ':9090'
});


//Set up robot objects
var settings = [];
var robotNames = ['deckard','roy','pris','zhora','leon'];
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
function consoleOut(str){
	var dt = new Date();
	var time = dt.getHours() + ":" + dt.getMinutes() + ":" + dt.getSeconds();
	jQuery("<code>[" + time + "] " + str + "</code> <br />").appendTo("#codeHistoryBody");
	jQuery("#codeHistoryBody").scrollTop(jQuery("#codeHistoryBody")[0].scrollHeight);
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
    	} else{
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

	var pos_obj = jQuery("#Position_obj");
	var pos_area = jQuery("#position_area");
	var moving = jQuery("#Movement");

	var certainties_obj = document.getElementById("obj_certainties");
	var targets_obj = document.getElementById("obj_targets");
	var positivities_obj = document.getElementById("obj_positivities");
	var object_relations = document.getElementById("obj_relations");
	var objects = document.getElementById("obj");
	var certainties_area = document.getElementById("area_certainties");
	var targets_area = document.getElementById("area_targets");
	var positivities_area = document.getElementById("area_positivities")
	var area_relations = document.getElementById("area_relation");
	var areas = document.getElementById("area");
	var certainties_mv = document.getElementById("mv_certainties");
	var targets_mv = document.getElementById("mv_targets");
	var positivities_mv = document.getElementById("mv_positivities")
	var movement_types = document.getElementById("mv_types");
	var movement_qualities = document.getElementById("mv_qualities");

	var tabs = [pos_obj, pos_area, moving];
	var certainty_cases = [certainties_obj, certainties_area, certainties_mv];
	var target_cases = [targets_obj, targets_area, targets_mv];
	var positivity_cases = [positivities_obj, positivities_area, positivities_mv];
	var discribers = [object_relations, area_relations, movement_types];
	var specifications = [objects, areas, movement_qualities];
	
	
	targets_obj.onchange = function(){
		if(targets_obj.value == "nothing"){
			positivities_obj.options[1].style.display="none";
		}else{
			positivities_obj.options[1].style.display="inline";
		}
	}

	targets_area.onchange = function(){
		if(targets_area.value == "nothing"){
			positivities_area.options[1].style.display="none";
		}else{
			positivities_area.options[1].style.display="inline";
		}
	}

	targets_mv.onchange = function(){
		if(targets_mv.value == "nothing"){
			positivities_mv.options[1].style.display="none";
		}else{
			positivities_mv.options[1].style.display="inline";
		}
	}

	movement_types.onchange = function(){
		if(movement_types.value == "stopped"){
			movement_qualities.style.display="none";
		}else{
			movement_qualities.style.display="inline";
		}
	}

	
	// Publish through ros every time 'submit' is pressed
	jQuery("#human_sensor_button").click(function() { 
	    
	    var s = [];
	    for(i = 0; i<tabs.length; i++){
	    	if(tabs[i].hasClass('active') == true){
	    		s[1] = certainty_cases[i];
			    s[2] = target_cases[i];
			    s[3] = positivity_cases[i];
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
	    
		var str = 	["" 		+ return_str[1],
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
})

jQuery('#start-stop').click(function(){
    jQuery(this).toggleClass("btn-success");
    jQuery(this).toggleClass("btn-danger");			        
    jQuery(this).children("span").toggleClass("glyphicon-play");
    jQuery(this).children("span").toggleClass("glyphicon-stop");			        
});