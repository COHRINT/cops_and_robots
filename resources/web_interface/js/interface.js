$("#deckard-visual").load(function() {
	console.log($(this).height($(this).contents().find("body").height()))
    $(this).height( $(this).contents().find("body").height() );
});

/**
* Set up primary elements to be manipulated by functions
*/

//Set up primary ROS master
// var rosurl =  'ws://fleming.recuv.org'; //commonly at 128.138.157.84
// var rosMaster = new ROSLIB.Ros({
//   url : rosurl + ':9000'
// });

var rosurl =  'ws://192.168.20.110'; //commonly at 128.138.157.84
var rosMaster = new ROSLIB.Ros({
  url : rosurl + ':9990'
});

// var rosurl =  'ws://127.0.0.1'; //commonly at 128.138.157.84
// var rosMaster = new ROSLIB.Ros({
//   url : rosurl + ':9090'
// });



//Set up robot objects
var settings = [];
var robotNames = ['deckard','roy','pris','zhora'];
var robotPorts = [':9091',':9092',':9093',':9094'];
var robots = [];

// First rostopic updates
var hover = false;
questionHovering(hover);
var cam = 'Deckard Visual';
humanViewState(cam);



/*
*NEED TO FIND OUT WHAT THIS DOES
*/
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
	  updateRobotQuestions(message.questions, message.weights, message.qids);
	});
}

/*
* Updates the robot question panel based on ROS messages
*/
function updateRobotQuestions(publishedQuestions, questionValues, questionIds){
	
	// Check if we get the right number and type/value of questions
	for(i = 0; i < questionValues.length; i++){
		if((questionValues.length != 5) || (questionValues[i] > 1) || (questionValues[i] < 0)){
			console.log("Not getting properly formed questions.");
			return;
		}
	}

	// Turn them all into precentages
	for(i = 0; i < questionValues.length; i++){
		questionValues[i] = questionValues[i] * 100;
	}

	var values = questionValues.slice();	

	//Send to pct_weights and robo_questions to index.php to be published in progress bars

	// Set text and weight for questions
	for(i = 0; i < 5; i++){
		$('#question_' + i + ' div p').html(publishedQuestions[i]);
		$('#question_' + i + ' div').attr("aria-valuenow", values[i].toString());
		$('#question_' + i + ' div').css("width", values[i].toString() + "%");
		$('#question_' + i + ' button.btn-success').attr("id", questionIds[i]);	
		$('#question_' + i + ' button.btn-danger').attr("id", questionIds[i]);
	}

	answeredQuestions(questionIds, publishedQuestions);
}

/*
* Sends information to ROS topic /answer and publishes to input history
*/
function answeredQuestions(questionIds, askedQuestions){

	jQuery('#robotQuestions div .btn-success').unbind().click(function() {
		var qid = this.id;
		var answeredQuestion;
		var bool = true;

	    for(i = 0; i < questionIds.length; i++){	
		    if (qid == questionIds[i]){
		    	answeredQuestion = askedQuestions[i];
		    	break;
		    }
		}
		var str = [answeredQuestion + " :YES"];
		consoleOut(str);
    	publishAnswer(qid, bool);
	});

	jQuery('#robotQuestions div .btn-danger').unbind().click(function() {
		var qid = this.id;
	    var answeredQuestion;
	    var bool = false;

        for(i = 0; i < questionIds.length; i++){	
    	    if (qid == questionIds[i]){
    	    	answeredQuestion = askedQuestions[i];
    	    	break;
    	    }
    	}
    	var str = [answeredQuestion + " :NO"];
    	consoleOut(str);
    	publishAnswer(qid, bool);
	});	
}

/**
* Select either Deckard's real camera or the control of any simulated robot
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
* Select either Deckard's real camera or the view of any simulated robot
* @param {String} robot
*/
function selectMap(robot){
    for (i = 0;  i < robotNames.length; i++){
    	if (robot === robots[i].nicename){
	    	robots[i].map = true;
    	} else{
	    	robots[i].map = false;
    	}
    }
    
    var str = "Probability map of " + robot + "." ;
    consoleOut(str);
    robotProbabilityMap(robot);
}

/**
* Publish which robot probability map should be projected to gazebo
*/
function robotProbabilityMap(str){

	rosMaster.on('connection', function() {
		console.log('Connected to websocket server.');
	});

		
	/*---------------------Codebook-------------------*/
	// Establish ROS topic for sending human messages
	var robot_probability_map = new ROSLIB.Topic({
	    ros : rosMaster,
	    name : '/robot_probability_map',
	    messageType : 'std_msgs/String'
	});
	
	var msg = new ROSLIB.Message({data: str}); 
	robot_probability_map.publish(msg);
}

/**
* Publish if mouse is on robot generated questions
*/
function questionHovering(bool){

	rosMaster.on('connection', function() {
	    console.log('Connected to websocket server.');
	});
	
	// Establish ROS topic for sending hover messages
	var questionHover = new ROSLIB.Topic({
	    ros : rosMaster,
	    name : '/question_hover',
	    messageType : 'std_msgs/Bool'
	});

    var msg = new ROSLIB.Message({data: bool});	 
	questionHover.publish(msg);
}

/**
* Publish human sensory data
*/
function humanInputSensor(str){

	rosMaster.on('connection', function() {
		console.log('Connected to websocket server.');
	});

		
	/*---------------------Codebook-------------------*/
	// Establish ROS topic for sending human messages
	var human_sensor = new ROSLIB.Topic({
	    ros : rosMaster,
	    name : '/human_sensor',
	    messageType : 'std_msgs/String'
	});
	
	var msg = new ROSLIB.Message({data: str}); 
	human_sensor.publish(msg);
}

/**
* Publish what camera the human is looking through
*/
function humanViewState(str){

	rosMaster.on('connection', function() {
		console.log('Connected to websocket server.');
	});
		
	/*---------------------Codebook-------------------*/
	// Establish ROS topic for sending human messages
	var human_view_state = new ROSLIB.Topic({
	    ros : rosMaster,
	    name : '/human_view_state',
	    messageType : 'std_msgs/String'
	});

	var msg = new ROSLIB.Message({data: str});	 
	human_view_state.publish(msg);
}

/**
* Publish the way the human answers the robots questions
*/
function publishAnswer(idNumber, bool){

	var idNum = parseInt(idNumber);

	rosMaster.on('connection', function() {
	    console.log('Connected to websocket server.');
	});
	
	// Establish ROS topic for sending hover messages
	var questionAnswer = new ROSLIB.Topic({
	    ros : rosMaster,
	    name : '/answer',
	    messageType : 'cops_and_robots_ros/Answer'
	});

    var msg = new ROSLIB.Message({qid: idNum, answer: bool});	
	questionAnswer.publish(msg);
}


/**
* Get settings parameters from UI and modify model appropriately
*/
function checkSettings(){
	//Select simulation vs. experiment

	//

 	var vicon = jQuery("#setting-source-vicon");
	var gazebo= jQuery("#setting-source-gazebo");
	var diffSettings = [vicon, gazebo];
	var vPorts =[":1234", ":1234", ":1234", ":1234"];
	var vTopics = ["/deckard/image", "/cam1/image", "/cam2/image", "/cam3/image"];
	var gPorts =[":1234", ":1234", ":1234", ":1234"];
	var gTopics = ["/deckard/camera/image_raw", "/security_camera1/camera/image_raw", "/security_camera2/camera/image_raw", "/security_camera3/camera/image_raw"]; 
	var ports = [vPorts, gPorts];
	var topics =[vTopics, gTopics];
	var ids = ["#deckard-visual", "#camera1-visual", "#camera2-visual", "#camera3-visual"];

    for(i = 0; i < diffSettings.length; i++){
    	if(diffSettings[i].parent().hasClass('active') == true){
	      for(j=0; j<ports[i].length; j++){
	      	// $(ids[j]).attr("src", "http://flemming.recov.org"+ports[i][j]+"/stream_viewer?topic="+topics[i][j]);
	      	//Regular topic stream
	      	$(ids[j]).attr("src", "http://192.168.20.110"+ports[i][j]+"/stream_viewer?topic="+topics[i][j]);

	  	  }
	  	  break;	 
		}
    }

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

	selectControl('Deckard');
	selectView('Deckard');	
	selectMap('Roy')
	
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
	
	
	// Takes out confusing options
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

	// Connects first three code boxes on the different human sensory tabs

	// may need to change how the first option is selected add to js
	// add in taget id matricies

	var targets = ('target_id_obj', 'target_id_area', 'target_id_mv');
	var m;
	for(i = 0; i < targetCases.length; i++){
		targetCases[i].onchange = function(){
			// issues start here
			for(j = 0; j < $('#targets[i]').length; j++){
				console.log($('#targets[i]').length);
				if (typeof($('#targets[i][j] option:selected')) !== 'undefined'){
					m = j;
					break;
					console.log(m);
				}
			}
			// if(i==0){
			// 	targets[1][m].attr("option ", "selected");
			// 	targets[2][m].attr("option ", "selected");
			// }else if(i==1){
			// 	targets[0][m].attr("option ", "selected");
			// 	targets[2][m].attr("option ", "selected");
			// }else{
			// 	targets[0][m].attr("option ", "selected");
			// 	targets[1][m].attr("option ", "selected");
			// }
		}
	}
	
	
	// Publish through ros every time 'submit' is pressed
	jQuery("#human_sensor_button").unbind().click(function() { 
	    
	    var s = [];
	    for(i = 0; i < tabs.length; i++){
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

		humanInputSensor(str);
		consoleOut(str);
	});
	       	    
	
	/*---------------------Teleoperation-------------------*/
	
	//Command switch
	jQuery('.ctrl-robot').unbind().click(function(e) {
	    e.preventDefault();
	    var robot = jQuery(this).parent().prev().text().trim();	    
	    selectControl(robot);
	})
	
	//View Switch
	jQuery('.view-robot').unbind().click(function(e) {
	    e.preventDefault();
	    var robot = jQuery(this).parent().prev().text().trim();	    
	    selectView(robot);
	})

	//Map Switch
	jQuery('.probability-robot').unbind().click(function(e) {
	    e.preventDefault();
	    var robot = jQuery(this).parent().prev().text().trim();	    
	    selectMap(robot);
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

	jQuery.get('js/typeahead.json', function(data){
	    jQuery("#chatInput").typeahead({ 
	    	source:data,
			items:'all',
			matcher: function (item) {
							        var last = this.query.split(' ');
							        this.query = jQuery.trim(last[last.length-1]);
							
							        if(this.query.length) return ~item.toLowerCase().indexOf(this.query.toLowerCase());
			},
			updater: function (item) {
								  	var input = this.$element.val().split(' ');
								  	for (var i=0; i < input.length-1; i++){
									  	word = input[i]
									  	if (item.toLowerCase().indexOf(word.toLowerCase()) > -1){
										  	item = item.replace(word + ' ','')
									  	}
									  	
								  	}
									return this.$element.val().replace(new RegExp(this.query + '$'),'') + item + ' ';
		    }
		 });
	},'json');
	
/*
	jQuery("#chatInput").change(function (){
		jQuery("#chatInput").typeahead({ 
	    	source:data,
			items:'all',
		 });

	});
*/
	$("#chatInput").keypress(function(event) {
	    if (event.which == 13) {
	        event.preventDefault();
    		var str = jQuery.trim(jQuery("#chatInput").val()) + '.'
		    humanInputSensor(str);
			consoleOut("You said: " + str);
			
			jQuery("#chatInput").val("")
	    }
	});
	
	// Publish through ros every time 'submit' is pressed
	jQuery("#human_chat_button").unbind().click(function() { 
		
		var str = jQuery.trim(jQuery("#chatInput").val()) + '.'
	    humanInputSensor(str);
		consoleOut("You said: " + str);
		
		jQuery("#chatInput").val("")
	});
	
}

/*
* Determines the view which the human is looking through
*/
function determineView(){
	//Determine what camera the human is looking through
	var viewTabs = ['#deckardVisual', '#prisVisual', '#royVisual', '#zhoraVisual', '#cam1', '#cam2', '#cam3'];
	var viewTabNames = ['Deckard Visual', 'Pris Visual', 'Roy Visual', 'Zhora Visual', 'Security Camera 1 Visual', 'Security Camera 2 Visual', 'Security Camera 3 Visual'];

	for(i = 0; i < viewTabs.length; i++){
		if(jQuery(viewTabs[i]).hasClass('active') == true){
			var str = viewTabNames[i];
			humanViewState(str);
			break;
		}
	}	
}

/**
* Uses ajax to call start bash script
*/
function callStartScript(arg){
    jQuery.ajax({ 
    	type: "POST",
        url: "cgi_bin/starting_script.php",
        dataType: "script",
        data:{setting: arg},  
        async: true,
        success: function(body){  
            alert('response received: ' + arg);              
        } 
    }); 

}
/**
* Uses ajax to call stop bash script
*/
function callStopScript(arg){
    jQuery.ajax({ 
    	type: "POST",
        url: "cgi_bin/stopping_script.php",
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
	robotUpdateSetup();
});

jQuery('#settings-save').unbind().click(function(){
	checkSettings();
	jQuery('#settings').modal('hide');
});

jQuery('.camera_views').unbind().click(function(){
	//wait .1 seconds before continuing otherwise wont set new element to active fast enough
	setTimeout(determineView, 100) 
})

jQuery('#robotQuestions').hover(function(){
    jQuery("div div.progress-bar").css("background-color", "#E6E6E6");
    var bool = true;
    questionHovering(bool);
	}, function(){
		jQuery("div div.progress-bar").css("background-color", "#A4ECFD");		
	var bool = false;
	questionHovering(bool);
});

jQuery('#start-stop').unbind().click(function(){

	var vicon = jQuery("#setting-source-vicon");
	var gazebo= jQuery("#setting-source-gazebo");
	var diffSettings = [vicon, gazebo];
	var activeSet = ['vicon', 'gazebo'];
	var setting = '';

    jQuery(this).toggleClass("btn-success");
    jQuery(this).toggleClass("btn-danger");			        
    jQuery(this).children("span").toggleClass("glyphicon-play");
    jQuery(this).children("span").toggleClass("glyphicon-stop");	

    for(i = 0; i<diffSettings.length; i++){
    	if(diffSettings[i].parent().hasClass('active') == true){
    		setting = activeSet[i];
    		break;
    	}
    }

	// Calls script function
    if(jQuery(this).children("span").hasClass("glyphicon-stop") == true){
    	callStartScript(setting);
    } else if (jQuery(this).children("span").hasClass("glyphicon-play") == true){
    	callStopScript(setting);
    }


});