/**
 * @author Russell Toris - rctoris@wpi.edu
 * edited for quad control by Matthew Aitken
 */

var KEYBOARDTELEOP = KEYBOARDTELEOP || {
  REVISION : '0.4.0-SNAPSHOT'
};

/**
 * @author Russell Toris - rctoris@wpi.edu
 */

/**
 * Manages connection to the server and all interactions with ROS.
 *
 * Emits the following events:
 *   * 'change' - emitted with a change in speed occurs
 *
 * @constructor
 * @param options - possible keys include:
 *   * ros - the ROSLIB.Ros connection handle
 *   * topic (optional) - the Twist topic to publish to, like '/cmd_vel'
 *   * throttle (optional) - a constant throttle for the speed
 */
KEYBOARDTELEOP.Teleop = function(options) {
  var that = this;
  options = options || {};
  var ros = options.ros;
  var topic = options.topic || '/cmd_vel';
  // permanent throttle
  var throttle = options.throttle || 1.0;

  // used to externally throttle the speed (e.g., from a slider)
  this.scale = 1.0;

  // linear x and y movement and angular z movement
  var x = 0;
  var y = 0;
  var z = 0;

  var cmdVel = new ROSLIB.Topic({
    ros : ros,
    name : topic,
    messageType : 'geometry_msgs/Twist'
  });

  // sets up a key listener on the page used for keyboard teleoperation
  var handleKey = function(keyCode, keyDown) {
    // used to check for changes in speed
    var oldX = x;
    var oldY = y;
    var oldZ = z;
    
    var pub = true;
	var z_ang = 0
    var speed = 0;
    // set speed a constant
    if (keyDown === true) {
      speed = throttle;
    }
    // check which key was pressed
    switch (keyCode) {
      case 16:
		// Move up
		z = 0.5 * speed;
      case 17:
		// move down
		z = -0.5 * speed;
      case 81:
        // turn left
        z_ang = 1 * speed;
        break;
      case 87:
        // forward
        x = 0.5 * speed;
        break;
      case 69:
        // turn right
        z_ang = -1 * speed;
        break;
      case 83:
        // backward
        x = -0.5 * speed;
        break;
      case 68:
        // strafe right
        y = -0.5 * speed;
        break;
      case 65:
        // strafe left
        y = 0.5 * speed;
        break;
      default:
        pub = false;    
    }

    // publish the command
    if (pub === true) {
      var twist = new ROSLIB.Message({
        angular : {
          x : 0,
          y : 0,
          z : z_ang
        },
        linear : {
          x : x,
          y : y,
          z : z
        }
      });
      cmdVel.publish(twist);
      console.log('Published:' + twist)

      // check for changes
      if (oldX !== x || oldY !== y || oldZ !== z) {
        that.emit('change', twist);
      }
    }
  };

  // handle the key
  var body = document.getElementsByTagName('body')[0];
  body.addEventListener('keydown', function(e) {
    handleKey(e.keyCode, true);
  }, false);
  body.addEventListener('keyup', function(e) {
    handleKey(e.keyCode, false);
  }, false);
};
KEYBOARDTELEOP.Teleop.prototype.__proto__ = EventEmitter2.prototype;
