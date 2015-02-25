<!-- PLEASE SEE AND UPDATE INSTRUCTIONS HERE: https://github.com/COHRINT/Cops-and-Robots/wiki/Operating-Procedures -->

<link href="sites/all/includes/interface.css" type="text/css" rel="stylesheet" />

<?php
$robots = array("Deckard","Roy","Zhora","Pris","Leon");
$map_objects = array("Wall 1","Wall 2","Wall 3","Wall 4","Wall 5","Wall 6","Wall 7","Wall 8")
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
			<h3 align="center">Deckard Visual</h3>		
<iframe  id="camera-visual" class="embed-responsive-item" src="http://fleming.recuv.org:8083/stream_viewer?topic=/camera/rgb/image_raw&width=547&height=400" height="416" width="555"  allowfullscreen="" frameborder="0"></iframe>

		</div><!-- /#visual -->
		
		<!-- Environment Map -->
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
<iframe class="embed-responsive-item" src="http://fleming.recuv.org:8080" height="416" width="555" allowfullscreen="" frameborder="0"></iframe>
			</div>
			
			<div class="modal fade bs-example-modal-lg" tabindex="-1" role="dialog" aria-labelledby="myLargeModalLabel" aria-hidden="true">
			  <div class="modal-dialog modal-lg" style="width: 100%;  height: 100%;  padding: 0;">
			    <div class="modal-content">
 <iframe src="http://fleming.recuv.org:8080" height="800px" width="100%" frameborder="0"></iframe>
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
			        <h3>Data Source</h3>
			        <div id="settings-source" class="btn-group" data-toggle="buttons">
					  <label class="btn btn-info active">
					    <input type="radio" name="options" id="setting-source-vicon" autocomplete="off" checked> Vicon
					  </label>
					  <label class="btn btn-info ">
					    <input type="radio" name="options" id="setting-source-sim" autocomplete="off" > Gazebo
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
		<div id="controls" class="col-md-6">
			<div>
			<button id="start-stop" type="button" class="btn btn-success btn-lg col-md-12">
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
					I 				
					<select size="6" id="sel1" class="form-control code-select"  >
					  <option selected="selected">think</option>
					  <option>know</option>				  
					</select>
					<select size="6" id="sel2" class="form-control code-select" >
					  <option selected="selected">nothing</option>	
					  <option>a robber</option>
			        <?php 
					foreach($robots as $name){  ?>
					  <option> <?php echo $name; ?></option>
					<?php } ?>
					</select>	
					is
					<select size="6" id="sel3" class="form-control code-select" >
					  <option  style="display:none" id="selMoving">moving</option>
					  <option selected="selected">behind</option>
					  <option>in front of</option>
					  <option>left of</option>
					  <option>right of</option>			 
<!--
					  <option>next to</option>				  
					  <option>near</option>
					  <option>far from</option>
-->
					</select>	
					<select size="6" id="sel4" class="form-control code-select" >
			        <?php 
			        $i = 0;
					foreach($map_objects as $obj){  
						$i++;
						if ($i == 1){ ?>
							<option selected="selected"><?php echo $obj; ?></option>
						<?php }else{?>
							  <option> <?php echo $obj; ?></option>
					<?php }} ?>
					</select>
					<select size="6" id="sel5" class="form-control code-select" style="display:none">
					  <option selected="selected">quickly</option>
					  <option>slowly</option>				  
					</select>	
					<select size="6" id="sel6" class="form-control code-select" style="display:none">
					  <option selected="selected">away</option>
					  <option>left</option>
					  <option>right</option>
					  <option>to you</option>
					</select>.
					<br />
					<div align="center" style="margin-top:10px"><button type="button" class="btn btn-success btn-lg" id="human_sensor_button" >Submit</button></div>								   
				  </div><!--/.panel-body-->
			</div>			
		</div>

		<div id="codeHistory" class="col-md-6">
			<div class="panel panel-default" style="margin-top:6px;">
				<div class="panel-heading">
			    	<h2 class="panel-title" align="center">Input History</h2>
				</div>
				<div class="panel-body" id="codeHistoryBody" style="height:183px; overflow-y:scroll;"></div>
			</div>
		</div>
	</div> <!-- /.row -->
</div> <!-- /.container -->

<script type="text/javascript" src="http://cdn.robotwebtools.org/EaselJS/current/easeljs.min.js"></script>
<script type="text/javascript" src="http://cdn.robotwebtools.org/EventEmitter2/current/eventemitter2.min.js"></script>
<script type="text/javascript" src="http://cdn.robotwebtools.org/mjpegcanvasjs/current/mjpegcanvas.min.js"></script>
<script type="text/javascript" src="http://cdn.robotwebtools.org/roslibjs/current/roslib.min.js"></script>
<script type="text/javascript" src="http://cdn.robotwebtools.org/ros2djs/current/ros2d.min.js"></script>
<script type="text/javascript" src="sites/all/includes/interface.js"></script>
<script type="text/javascript" src="sites/all/includes/jquery-cookie/jquery.cookie.js"
