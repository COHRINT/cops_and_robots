<!-- PLEASE SEE AND UPDATE INSTRUCTIONS HERE: https://github.com/COHRINT/Cops-and-Robots/wiki/Operating-Procedures -->

<link href="interface.css" type="text/css" rel="stylesheet" />
<link href="bootstrap-3.3.4-dist/bootstrap.min.css" rel="stylesheet">

<?php
$robots = array("Deckard", "Pris", "Roy", "Zhora");
$targets = array("nothing", "a robber", "Roy", "Pris", "Zhora");
$certainties = array("I think", "I know");
$positivities = array("is", "is not");
$object_relations = array("behind", "infront of", "left of", "right of");
$objects = array("Deckard", "the bookshelf", "the chair", "the desk", "the table");
$area_relations = array("inside", "near", "outside");
$areas = array("the study", "the billiard room", "the hallway", "the dining room", "the kitchen", "the library");
$movement_types = array("moving", "stopped");
$movement_qualities = array("slowly", "moderately", "quickly");

$pos_obj = array($certainties, $targets, $positivities, $object_relations, $objects);
$Id_obj = array("obj_certainties","obj_targets", "obj_positivities", "obj_relations", "obj");
$pos_area = array($certainties, $targets, $positivities, $area_relations, $areas);
$Id_area = array("area_certainties","area_targets", "area_positivities", "area_relation", "area");
$move = array($certainties, $targets, $positivities, $movement_types, $movement_qualities);
$Id_move = array("mv_certainties","mv_targets", "mv_positivities", "mv_types", "mv_qualities");
?>

<div class="container">
	<div class="row">
		<!-- Control Alert -->
		<div id="alert" class="alert alert-info alert-dismissable col-md-12">
		  <button type="button" class="close" data-dismiss="alert" aria-hidden="true">&times;</button>
		  <h3>Controls</h3>
		  	<p><pre>
. -> speed up   w -> move forward   a -> rotate counterclockwise  q -> turn left   space -> stop            Hit buttons once
, -> slow down  s -> move backward  d -> rotate clockwise         e -> turn right                     instead of holding them down
			</pre></p>
		</div><!-- /#alert -->


		<!-- Camera Visual -->
		<div id="visual" class="col-md-6">
		<!-- Camera Options -->
		        <div role="tabpanel">



	                <!-- Nav tabs -->
            		<ul class="nav nav-tabs" role="tablist">
                   		 	<li role="presentation" class="active"><a href="#Deckard's Camera" aria-controls="Deckard's Camera" data-toggle="tab">Deckard's Camera</a></li>
                   		 	<li class="dropdown">
                   		 		<a href="#Robber Cameras" class="dropdown-toggle" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">Robber Cameras<span class="caret"></span></a>
                   		 			 <ul class="dropdown-menu">
		                   		 		<li><a data-toggle="tab" href="#Pris's Camera" aria-controls="Pris's Camera" >Pris's Camera</a></li>
		                   		 		<li><a data-toggle="tab" href="#Roy's Camera" aria-controls="Roy's Camera" >Roy's Camera</a></li>
		                   		 		<li><a data-toggle="tab" href="#Zhora's Camera" aria-controls="Zhora's Camera" >Zhora's Camera</a></li>
                   		 			</ul>
                   		 	</li>
                   		 	<li class="dropdown">
                   		 		<a href="#Security Cameras" class="dropdown-toggle" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">Security Cameras<span class="caret"></span></a>
                   		 			 <ul class="dropdown-menu">
		                    			<li><a data-toggle="tab" href="#Camera 1" aria-controls="Camera 1" >Camera 1</a></li>
		                    			<li><a data-toggle="tab" href="#Camera 2" aria-controls="Camera 2" >Camera 2</a></li>
		                    			<li><a data-toggle="tab" href="#Camera 3" aria-controls="Camera 3" >Camera 3</a></li>
		                    		</ul>
		                    </li>
            		</ul>

		  			<!-- Tab panes -->
	  				<div class="tab-content">
	    					<div class="tab-pane active" id="Deckard's Camera"> 
	    						<iframe  id="deckard-camera-visual" class="embed-responsive-item" src="" height="416" width="555"  allowfullscreen="" frameborder="0"></iframe> 
	    					</div>
	    					<div class="tab-pane" id="Pris's Camera"> 
	    						<iframe  id="pris-camera-visual" class="embed-responsive-item" src="" height="416" width="555"  allowfullscreen="" frameborder="0"></iframe> 
	    					</div>
	    					<div class="tab-pane" id="Roy's Camera"> 
	    						<iframe  id="roy-camera-visual" class="embed-responsive-item" src="" height="416" width="555"  allowfullscreen="" frameborder="0"></iframe> 
	    					</div>
	    					<div class="tab-pane" id="Zhora's Camera"> 
	    						<iframe  id="zhora-camera-visual" class="embed-responsive-item" src="" height="416" width="555"  allowfullscreen="" frameborder="0"></iframe> 
	    					</div>
	    					<div class="tab-pane" id="Camera 1"> 
	    						<iframe  id="camera1-visual" class="embed-responsive-item" src="" height="416" width="555"  allowfullscreen="" frameborder="0"></iframe> 
	    					</div>
	    					<div class="tab-pane" id="Camera 2"> 
	    						<iframe  id="camera2-visual" class="embed-responsive-item" src="" height="416" width="555"  allowfullscreen="" frameborder="0"></iframe> 
	    					</div>
	    					<div class="tab-pane" id="Camera 3"> 
	    						<iframe  id="camera3-visual" class="embed-responsive-item" src="" height="416" width="555"  allowfullscreen="" frameborder="0"></iframe> 
	    					</div>
	  				</div>

				</div>

		</div><!-- /#visual -->

		<!-- Environment Map -->
			<!-- Currently running through local host -->

		
		<div id="map-test" class="col-md-6">
			<h3 align="center">Environment Map
				<button class="btn btn-info btn-sm" data-toggle="modal" data-target=".bs-example-modal-lg">
					<span class="glyphicon glyphicon-resize-full"></span>
				</button>
				<button type="button" class="btn btn-info btn-sm" data-toggle="modal" data-target="#settings">
					<span class="glyphicon glyphicon-wrench"></span>
				</button>
			</h3>
			<div class="embed-responsive embed-responsive-4by3">
				<iframe class="embed-responsive-item" src="http://localhost:8080" height="416" width="555" allowfullscreen="" frameborder="0"></iframe>
			</div>

			<div class="modal fade bs-example-modal-lg" tabindex="-1" role="dialog" aria-labelledby="myLargeModalLabel" aria-hidden="true">
			  <div class="modal-dialog modal-lg" style="width: 100%;  height: 100%;  padding: 0;">
			    <div class="modal-content">
 					<iframe src="http://localhost:8080" height="800px" width="100%" frameborder="0"></iframe>
			    </div>
			  </div>
			</div>

			<!-- Settings -->
			<div class="modal fade" id="settings" tabindex="-1" role="dialog" aria-labelledby="myModalLabel" aria-hidden="true">
			  <div class="modal-dialog">
			    <div class="modal-content">
			      <div class="modal-header">
			        <button type="button" class="close" data-dismiss="modal"><span aria-hidden="true">&times;</span><span class="sr-only">Close</span></button>
			        <h4 class="modal-title" id="myModalLabel">Settings</h4>
			      </div>
			      <div class="modal-body">
			      	
			      	<h3>Search Style</h3>
			        <div id="settings-source" class="btn-group" data-toggle="buttons">
					  <label class="btn btn-info active">
					    <input type="radio" name="options" id="setting-style-static" autocomplete="off" checked> Static
					  </label>
					  <label class="btn btn-info ">
					    <input type="radio" name="options" id="setting-style-dynamic" autocomplete="off" > Dynamic
					  </label>
					</div>


			        <h3>Experiment/Simulation Type</h3>
			        <div id="settings-source" class="btn-group" data-toggle="buttons">
					  <label class="btn btn-info active">
					    <input type="radio" name="options" id="setting-source-vicon" autocomplete="off" checked > Vicon
					  </label>
					  <label class="btn btn-info ">
					    <input type="radio" name="options" id="setting-source-gazebo" autocomplete="off" > Gazebo
					  </label>
					  <label class="btn btn-info ">
					    <input type="radio" name="options" id="setting-source-python" autocomplete="off" > Python
					  </label>
					</div>

			        <h3>Active Agents</h3>
					<div class="btn-group" data-toggle="buttons">
			        <?php
					foreach($robots as $name){  ?>
					  <label id="setting-<?php echo strtolower($name) ?>-active" class="btn btn-info active">
					    <input type="checkbox" autocomplete="off" checked> <?php echo $name; ?>
					  </label>
					<?php } ?>
					</div>
			      </div>
			      <div class="modal-footer">
			        <button type="button" class="btn btn-default" data-dismiss="modal">Close</button>
			        <button id="settings-save" type="button" class="btn btn-primary">Save changes</button>
			      </div>
			    </div>
			  </div>
			</div>
		</div><!-- /#map -->

		<!-- Controls -->
		<!-- Major Lanch Button -->
		<!-- onclick= "<?php //shell_exec("/bash/start_stop.sh");?>" -->


		<div id="controls" class="col-md-6">
			<div>
			<button id="start-stop" type="button" class="btn btn-success btn-lg col-md-12" >
				<span class="glyphicon glyphicon-play"></span>
			</button>
			</div>
		</div><!-- /#controls -->

		<!-- Robot Buttons -->
		<div id="robots" class="col-md-6" style="text-align:center">
			<?php
			foreach($robots as $name){  ?>
			<div class="btn-group">
			<!--ADD TOOLTIP! -->
			  <button id="<?php echo strtolower($name) ?>-button" type="button" class="btn btn-lg btn-default dropdown-toggle" data-toggle="dropdown" aria-expanded="false">
			    <?php echo $name; ?> <span class="caret"></span>
			  </button>
			  <ul class="dropdown-menu" role="menu">
			    <li class="view-robot"><a href="#">View</a></li>
			    <li class="ctrl-robot"><a href="#">Control</a></li>
			  </ul>
			</div>
			<?php } ?>
			</div>
		</div> <!-- #robots -->

		<!-- Code Box -->
		<br clear="all">
		<div id="codeBox" class="col-md-6">

			<div class="panel panel-default" style="margin-top:6px;">
				<div class="panel-heading">
			    	<h2 class="panel-title" align="center">Human Sensory Input</h2>
				</div>

				<div class="panel-body form-inline" id="code" >

				   <div role="tabpanel">

		                <!-- Nav tabs -->
                		<ul class="nav nav-tabs" role="tablist">
                       		 	<li role="presentation" class="active"><a href="#Position_obj" aria-controls="Position_obj" role="tab" data-toggle="tab">Position (Object)</a></li>
                        		<li role="presentation"><a href="#position_area" aria-controls="position_area" role="tab" data-toggle="tab">Position (Area)</a></li>
                        		<li role="presentation"><a href="#Movement" aria-controls="Movement" role="tab" data-toggle="tab">Movement</a></li>
                		</ul>

				  		<!-- Tab panes -->
				  				<style>
								 .bloc { display:inline-block; vertical-align:top; overflow:hidden; border:solid grey 0px; }
								 .bloc select { padding:15px; margin:-5px -20px -5px -5px; }
								</style>	

		  				<div class="tab-content">

		    				<div id='Position_obj' class='tab-pane active'>
		    					<?php 
		    						for($i = 0; $i < count($pos_obj); $i++){
		    							PositionViaObject($pos_obj[$i], $Id_obj[$i]);
		    						}
		    					?>
	    					</div> <!-- Pos_obj -->

	    					<div id='position_area' class='tab-pane'>
		    					<?php 
		    						for($i = 0; $i < count($pos_area); $i++){
		    							PositionViaArea($pos_area[$i], $Id_area[$i]);
		    						}
		    					?>
	    					</div> <!-- Pos_area -->

	    					<div id='Movement' class='tab-pane'>
		    					<?php 
		    						for($i = 0; $i < count($move); $i++){
		    							Velocity($move[$i], $Id_move[$i]);
		    						}
		    					?>
	    					</div> <!-- Movement -->
		  				
		  				</div> <!-- Tab Content -->
					</div><!-- Tabbing -->
					<br />
					<div align="center" style="margin-top:10px"><button type="button" class="btn btn-success btn-lg" id="human_sensor_button" >Submit</button></div>								   
				</div><!--/.panel-body-->
			</div>
		</div><!-- /#codeBox -->

		<div id="codeHistory" class="col-md-6">
			<div class="panel panel-default" style="margin-top:6px;">
				<div class="panel-heading">
			    	<h2 class="panel-title" align="center">Input History</h2>
				</div>
				<div class="panel-body" id="codeHistoryBody" style="height:183px; overflow-y:scroll;"></div>
			</div><!-- /.panel -->
		</div><!-- /.codeHistory -->
	</div> <!-- /.row -->
</div> <!-- /.container -->


<!-- PHP function -->
	<?php function PositionViaObject(array $pos_obj, $Id_obj){ ?>
		<div class="bloc">
		<select size="6" id="<?php echo $Id_obj ?>" class="form-control code-select"  >
		<?php for($i = 0; $i < count($pos_obj); $i++){ ?>
			<option> <?php echo $pos_obj[$i]; ?> </option>
		<?php } ?>
			</select>
		</div> <!-- Scrollbar blocker --> 
	<?php } ?>

	<?php function PositionViaArea(array $pos_area, $Id_area){ ?>
		<div class="bloc">
			<select size="6" id="<?php echo $Id_area ?>" class="form-control code-select"  >
		<?php for($i = 0; $i < count($pos_area); $i++){ ?>
			<option> <?php echo $pos_area[$i]; ?> </option>
		<?php } ?>
			</select>
		</div> <!-- Scrollbar blocker --> 
	<?php } ?>

	<?php function Velocity(array $move, $Id_move){ ?>
		<div class="bloc">
			<select size="6" id="<?php echo $Id_move ?>" class="form-control code-select"  >
		<?php for($i = 0; $i < count($move); $i++){ ?>
			<option> <?php echo $move[$i]; ?> </option>
		<?php } ?>
			</select>
		</div> <!-- Scrollbar blocker --> 
	<?php } ?>






<script type="text/javascript" src="http://cdn.robotwebtools.org/EaselJS/current/easeljs.min.js"></script>
<script type="text/javascript" src="http://cdn.robotwebtools.org/EventEmitter2/current/eventemitter2.min.js"></script>
<script type="text/javascript" src="http://cdn.robotwebtools.org/mjpegcanvasjs/current/mjpegcanvas.min.js"></script>
<script type="text/javascript" src="http://cdn.robotwebtools.org/roslibjs/current/roslib.min.js"></script>
<script type="text/javascript" src="http://cdn.robotwebtools.org/ros2djs/current/ros2d.min.js"></script>
<script src="https://ajax.googleapis.com/ajax/libs/jquery/1.11.2/jquery.min.js"></script>
<script src="bootstrap-3.3.4-dist/bootstrap.min.js"></script>
<script type="text/javascript" src="js/keyboardteleopquadrotor.js"></script>
<script type="text/javascript" src="js/interface.js"></script>


