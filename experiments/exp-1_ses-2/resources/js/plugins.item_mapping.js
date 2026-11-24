// String.format prototype
// as per fearphage on https://stackoverflow.com/a/4673436 (last accessed 28/12/2022 11:59)
if (!String.prototype.format) {
    String.prototype.format = function() {
      var args = arguments;
      return this.replace(/{(\d+)}/g, function(match, number) { 
        return typeof args[number] != 'undefined'
          ? args[number]
          : match
        ;
      });
    };
}

// shake animation
function shakeWord($element) {
    $element.addClass('shake');

    // Remove the class after animation ends
    $element.one('animationend webkitAnimationEnd oAnimationEnd MSAnimationEnd', function () {
        $(this).removeClass('shake');
    });
}

// sound effects
function playSound(url) {
    const audio = new Audio(url);
    audio.play();
}

// Separate function for path animation
function animateParticleRandomFlight($particle, x_i, y_i, t_x, t_y, duration = 600) {
    const start = performance.now();

    // Control point for curved motion
    const controlX = x_i + (Math.random() - 0.5) * 150;
    const controlY = y_i - 50 - Math.random() * 50;

    function easeInOutQuad(t) {
        return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
    }

    function animate(time) {
        const elapsed = time - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased = easeInOutQuad(progress);

        // Quadratic Bézier curve (start → control → end)
        const x = (1 - eased) * (1 - eased) * x_i +
                  2 * (1 - eased) * eased * controlX +
                  eased * eased * t_x;

        const y = (1 - eased) * (1 - eased) * y_i +
                  2 * (1 - eased) * eased * controlY +
                  eased * eased * t_y;

        $particle.css({ left: x + 'px', top: y + 'px' });

        if (progress < 1) {
            requestAnimationFrame(animate);
        } else {
            $particle.remove();
        }
    }

    requestAnimationFrame(animate);
}

// word categorisation in a circle of images plugin
jsPsych.plugins['item_mapping'] = (function(){
    // preloads
    jsPsych.pluginAPI.registerPreload('item_mapping', 'sound_fx_wheesh', 'audio');
    jsPsych.pluginAPI.registerPreload('item_mapping', 'sound_fx_whoosh', 'audio');
    jsPsych.pluginAPI.registerPreload('item_mapping', 'sound_fx_up', 'audio');
    jsPsych.pluginAPI.registerPreload('item_mapping', 'sound_fx_down', 'audio');
    jsPsych.pluginAPI.registerPreload('item_mapping', 'sound_fx_error', 'audio');
    jsPsych.pluginAPI.registerPreload('item_mapping', 'sound_fx_success', 'audio');
    jsPsych.pluginAPI.registerPreload('item_mapping', 'sound_fx_clock', 'audio');
    jsPsych.pluginAPI.registerPreload('item_mapping', 'sound_fx_alarm', 'audio');

    // language specs
    const lang = {
        'de': {
            'loading': 'Aufgabe wird geladen...',
            'board_this': 'Dieser Kreis',
            'board_all': 'Alle Kreise',
            'audio_on': 'Ton an',
            'audio_off': 'Ton aus',
            'time': 'Zeit',
            'time_up': 'Zeit ist um!\nBitte zum Ende kommen.',
            'break': 'Du kannst nun eine kurze Pause nehmen. Drücke auf `Fertig`, um fortzufahren.',
            'complete': 'Fertig',
            'complete_in': 'möglich in',
            'confirmation': 'Sind Sie sich sicher?'
        },
        'en': {
            'loading': 'Loading task...',
            'board_this': 'Current board',
            'board_all': 'All boards',
            'audio_on': 'Sound on',
            'audio_off': 'Sound off',
            'time': 'Time',
            'time_up': 'Time\'s up!\nPlease finish up the board.',
            'break': 'You may take a short break. Press `Done` to continue.',
            'complete': 'Done',
            'complete_in': 'possible in',
            'confirmation': 'Are you sure?'
        }
    }

    // plugin specification
    var plugin = {};
    plugin.info = {
        name: 'word_categorisation_circle',
        parameters: {
            no: {
                type: jsPsych.plugins.parameterType.INT,
                default: 0,
                pretty_name: 'Number',
                description: 'Trial number.'
            },
            block: {
                type: jsPsych.plugins.parameterType.INT,
                default: 0,
                pretty_name: 'Block',
                description: 'Block number.'
            },
            anchors: {
                type: jsPsych.plugins.parameterType.COMPLEX,
                default: [],
                pretty_name: 'Anchors',
                description: 'A list of anchors that participants should arrange on the circle.'
            },
            words: {
                type: jsPsych.plugins.parameterType.COMPLEX,
                default: [],
                pretty_name: 'Item list',
                description: 'A list of strings that participants should arrange on the circle.'
            },
            per_circle: {
                type: jsPsych.plugins.parameterType.INT,
                default: 20,
                pretty_name: 'Items per circle',
                description: 'Number of items per circle.'
            },
            max_attempts: {
                type: jsPsych.plugins.parameterType.INT,
                default: 100,
                pretty_name: 'Maximum attempts',
                description: 'Maximum number of attempts to place an item on the circle.'
            },
            max_circles: {
                type: jsPsych.plugins.parameterType.INT,
                default: 10,
                pretty_name: 'Maximum circles',
                description: 'Maximum number of circles.'
            },
            circles: {
                type: jsPsych.plugins.parameterType.COMPLEX,
                default: [],
                pretty_name: 'Circles',
                description: 'A list of circles.'
            },
            identities: {
                type: jsPsych.plugins.parameterType.COMPLEX,
                default: [],
                pretty_name: 'Circle identities',
                description: 'A list of identifiers for the circles (useful, for example, for repetitions).'
            },
            skip_confirmation: {
                type: jsPsych.plugins.parameterType.BOOLEAN,
                default: true,
                pretty_name: 'Skip confirmation',
                description: 'Whether to skip the confirmation step.'
            },
            freeze_words: {
                type: jsPsych.plugins.parameterType.BOOLEAN,
                default: false,
                pretty_name: 'Freeze words',
                description: 'Whether to freeze the words on the circle.'
            },
            spawn_outside: {
                type: jsPsych.plugins.parameterType.BOOLEAN,
                default: true,
                pretty_name: 'Spawn outside',
                description: 'Whether to spawn items outside the circle.'
            },
            gamify: {
                type: jsPsych.plugins.parameterType.BOOLEAN,
                default: false,
                pretty_name: 'Gamify',
                description: 'Whether to gamify the task.'
            },
            sound: {
                type: jsPsych.plugins.parameterType.BOOLEAN,
                default: false,
                pretty_name: 'With sound',
                description: 'Whether to play sound effects when an item is placed on the circle.'
            },
            sound_fx_wheesh: {type: jsPsych.plugins.parameterType.AUDIO, default: './resources/sounds/wheesh.mp3'},
            sound_fx_whoosh: {type: jsPsych.plugins.parameterType.AUDIO, default: './resources/sounds/whoosh.mp3'},
            sound_fx_up: {type: jsPsych.plugins.parameterType.AUDIO, default: './resources/sounds/slide.mp3'},
            sound_fx_down: {type: jsPsych.plugins.parameterType.AUDIO, default: './resources/sounds/tap.mp3'},
            sound_fx_error: {type: jsPsych.plugins.parameterType.AUDIO, default: './resources/sounds/error.mp3'},
            sound_fx_success: {type: jsPsych.plugins.parameterType.AUDIO, default: './resources/sounds/success.mp3'},
            sound_fx_clock: {type: jsPsych.plugins.parameterType.AUDIO, default: './resources/sounds/clock.mp3'},
            sound_fx_alarm: {type: jsPsych.plugins.parameterType.AUDIO, default: './resources/sounds/alarm.mp3'},
            min_per_circle: {
                type: jsPsych.plugins.parameterType.INT,
                default: 2.5 * 60 * 1000,
                pretty_name: 'Minimum time per circle',
                description: 'Minimum time that should be taken per circle in milliseconds.'
            },
            time_per_circle: {
                type: jsPsych.plugins.parameterType.INT,
                default: 5 * 60 * 1000,
                pretty_name: 'Time per circle',
                description: 'Time per circle in milliseconds.'
            },
            circles_per_break: {
                type: jsPsych.plugins.parameterType.INT,
                default: 0,
                pretty_name: 'Circles per break',
                description: 'Circles to complete before break is offered, or zero to deactivate.'
            },
            study_language: {
                type: jsPsych.plugins.parameterType.STRING,
                default: 'en',
                pretty_name: 'Experiment language',
                description: 'Experiment language.'
            },
            sampling_frequency: {
                type: jsPsych.plugins.parameterType.INT,
                default: 10,
                pretty_name: 'Sampling frequency',
                description: 'Sampling frequency for mouse positions.'
            }
        }
    }

    // trial logic
    plugin.trial = function(e, t) {
        // log start of trial
        console.log("Trial", t, ": Starting...");

        // setup data structure
        let data = {
            no: t.no,
            block: t.block,
            items: t.items,
            boards: {
                'initial': [],
                'actions': [],
                'final': [],
                'rt': []
            },
            identities: [],
            rt: null
        };

        let vocabulary = {};

        // setup identities quickly
        data.identities = t.identities.map(String).join(',');

        // Utility: Randomly sample k words from an array using Fisher–Yates shuffling.
        let sampleCircle = function(words, k) {
            const shuffled = words.slice();
            for (let i = shuffled.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
            }
            return shuffled.slice(0, k);
        };
        
        // Utility: Given an array of words (a circle), get all unique pairs from it.
        // Each pair is represented as a string 'wordA|wordB' where wordA < wordB alphabetically.
        let getPairs = function(circle) {
            const pairs = new Set();
            for (let i = 0; i < circle.length; i++) {
                for (let j = i + 1; j < circle.length; j++) {
                    const pair = [circle[i], circle[j]].sort().join('|');
                    pairs.add(pair);
                }
            }
            return pairs;
        };
        
        // Main function: Greedily sample circles until every pair from all_words appears in at least one circle.
        let getCoveringCircles = function(all_words, circleSize, iterationsPerCircle = 1000) {
            // Initialize the set of all pairs of words from all_words.
            const uncoveredPairs = new Set();
            for (let i = 0; i < all_words.length; i++) {
                for (let j = i + 1; j < all_words.length; j++) {
                    const pair = [all_words[i], all_words[j]].sort().join('|');
                    uncoveredPairs.add(pair);
                }
            }
        
            const circles = [];
            
            // Continue until all pairs have been covered.
            while ((uncoveredPairs.size > 0) & (circles.length < t.max_circles)) {
                let bestCircle = null;
                let bestNewPairs = new Set(); // pairs that are covered newly by the best circle
            
                // Try several random circles and pick the one that covers the most uncovered pairs.
                for (let iter = 0; iter < iterationsPerCircle; iter++) {
                    const circle = sampleCircle(all_words, circleSize);
                    const circlePairs = getPairs(circle);
                    
                    // Count new pairs that this circle would cover.
                    const newPairs = new Set();
                    circlePairs.forEach(pair => {
                        if (uncoveredPairs.has(pair)) {
                            newPairs.add(pair);
                        }
                    });
                    
                    if (newPairs.size > bestNewPairs.size) {
                        bestCircle = circle;
                        bestNewPairs = newPairs;
                    }
                }
                
                // If no improvement found (which is unlikely in the early iterations), warn and break.
                if (!bestCircle) {
                    console.error("No additional coverage could be achieved with the current search.");
                    break;
                }
                
                circles.push(bestCircle);
                
                // Remove the pairs covered by the best circle from the set of uncovered pairs.
                bestNewPairs.forEach(pair => uncoveredPairs.delete(pair));
                
                console.log(`Added a circle; remaining uncovered pairs: ${uncoveredPairs.size} (${circles.length})`);
            }

            // finally, let's check for coverage
            let coverage = [];

            for (let i = 0; i < circles.length; i++) {
                for (let j = 0; j < circles[i].length; j++) {
                    if (!coverage.includes(circles[i][j])) {
                        coverage.push(circles[i][j]);
                    }
                }
            }

            // determine any missing items
            let missing = 0;

            for (let i = 0; i < t.words.length; i++) {
                if (!coverage.includes(t.words[i])) {
                    missing += 1;
                }
            }

            console.log(`Missing words: ${missing} (${missing / t.words.length})`);

            // as a fallback, we will rerun the algorithm
            if (missing > 0) {
                console.log("Re-running with more circles...");
                return getCoveringCircles(t.words, circleSize, iterationsPerCircle);
            }
            
            return circles;
        };

        let run_trial = function(current_trial){
            // grab items
            let items = circles[current_trial];

            // clear canvas
            e.innerHTML = '';

            // setup px conversion
            let pixelsToCoord = function(x, y, z, r){
                let nx = ((x - z[1]) / r);
                let ny = ((y - z[0]) / r) * (-1);

                return [nx, ny];
            };

            // setup json storage
            let trial_initial = {};
            let trial_final = {};
            
            // setup trial
            let pTrial = function(){
                // clear container
                e.innerHTML = '';

                // window setup
                const h = window.innerHeight;
                const w = window.innerWidth;

                // scale circle accordingly
                let p = Math.min(h / 5, w / 5);
                let d = 0.95 * Math.min(h - p, w - p);
                let r = d / 2;
                let r2 = r * 1.0;
                let Z = [h / 2, w / 2];
                let delta = 2 * Math.PI / items.length;
                let theta = 2 * Math.PI;

                // setup canvas
                let trialContainer = $(
                    '<div id="circle-header" class="circle-header">' +
                      '<div class="progress-container">' +
                        '<div class="progress-bar">' +
                          '<div id="progress-time" class="progress-fill time-fill" style="width: 0%;"></div>' +
                          '<span class="progress-label" id="progress-time-label">' + lang[t.study_language].time + '</span>' +
                        '</div>' +
                        '<div class="progress-bar">' +
                          '<div id="progress-this" class="progress-fill" style="width: 0%;"></div>' +
                          '<span class="progress-label">' + lang[t.study_language].board_this + '</span>' +
                        '</div>' +
                        '<div style="width: 50px;"></div>' + 
                        '<div class="progress-bar">' +
                          '<div id="progress-all" class="progress-fill" style="width: 0%;"></div>' +
                          '<span class="progress-label">' + lang[t.study_language].board_all + '</span>' +
                        '</div>' +
                        '<div class="sound-toggle-wrapper" style="margin-top: -8px;">' +
                            '<input type="checkbox" id="mute-toggle" />' +
                            '<label for="mute-toggle" class="sound-toggle"></label>' +
                            '<span id="mute-label">' + lang[t.study_language].audio_on + '</span>' +
                        '</div>' + 
                      '</div>' +
                    '</div>' +
                    '<fieldset style="border: none;">' +
                      '<div id="circle-task" style="width: 100% !important; height: 100% !important; user-select: none;"></div>' +
                    '</fieldset>'
                  );
                let taskContainer = trialContainer.find('#circle-task');
                let canvas = $('<canvas id="circle-context" width="{0}" height="{1}"></canvas>'.format(d+3, d+3));
                let submit = $('<div style="position: absolute !important; top 10px; left: 0; right: 0; margin: auto;"><button id="circle-task-submit" class="jspsych-btn fancy-button" disabled>' + lang[t.study_language].complete + '</button></div>');
                canvas.appendTo(taskContainer);
                submit.appendTo(taskContainer);
                trialContainer.appendTo(e);

                // setup tracking of individual actions and reaction times
                let actions = [];
                let actions_start_time = performance.now();
                let mouse_last_event = { x: 0, y: 0, t: 0 };
                
                // setup tracking of mouse movements
                let mouse_listener = $(document).on(
                    'mousemove.cursor', 
                    function(event){
                        // update cached position
                        mouse_last_event = { 
                            x: event.clientX,
                            y: event.clientY,
                            t: performance.now() - actions_start_time
                        };
                    }
                );

                let mouse_recorder = function(){
                    // grab last event
                    const centerX = mouse_last_event.x;
                    const centerY = mouse_last_event.y;

                    // bring into coordinate space
                    let xy = pixelsToCoord(centerX, centerY, Z, r);
                    
                    // log event
                    actions.push({ type: 'cursor', pos: xy, time: performance.now() - actions_start_time });
                };

                let mouse_interval = setInterval(mouse_recorder, 1000 / t.sampling_frequency);
                
                // setup sound controls
                let soundToggle = $('#mute-toggle');
                let soundLabel = $('#mute-label');
                soundToggle.on('click', function() {
                    if (soundToggle.prop('checked')) {
                        soundToggle.prop('checked', true);
                        soundLabel.text(lang[t.study_language].audio_on);
                        t.sound = true;
                    } else {
                        soundToggle.prop('checked', false);
                        soundLabel.text(lang[t.study_language].audio_off);
                        t.sound = false;
                    }
                });

                // initialise sound controls
                if (t.sound) {
                    soundToggle.prop('checked', true);
                    soundLabel.text(lang[t.study_language].audio_on);
                } else {
                    soundToggle.prop('checked', false);
                    soundLabel.text(lang[t.study_language].audio_off);
                }

                // update progress (all)
                let progress = current_trial / (circles.length - 1) * 100;
                $('#progress-all').css('width', `${progress}%`);
                
                // create circle animation
                function expandCircle(duration = 1000, fps = 60) {
                    const canvas = $('#circle-context').get(0);
                    const context = canvas.getContext("2d");
                    const r_full = canvas.width / 2 - 1;
                    const centerX = r_full + 1;
                    const centerY = r_full + 1;
                
                    let frame = 0;
                    const totalFrames = duration / (1000 / fps);
                
                    function animate() {
                        const progress = frame / totalFrames;
                        const r = r_full * progress;
                
                        // Clear the canvas
                        context.clearRect(0, 0, canvas.width, canvas.height);
                
                        // Draw smaller circle
                        context.beginPath();
                        context.arc(centerX, centerY, r+1, 0, 2 * Math.PI);
                        context.fillStyle = `rgb(200, 200, 200)`;
                        context.fill();
                
                        frame++;
                        if (frame <= totalFrames) {
                            requestAnimationFrame(animate);
                        }
                    }
                
                    animate();
                }

                if (t.gamify) {
                    if (t.sound) playSound(t.sound_fx_whoosh);
                    expandCircle(duration = 500, fps = 60);
                } else {
                    if (t.sound) playSound(t.sound_fx_whoosh);
                    expandCircle(duration = 1, fps = 60);
                }

                // setup individual items
                items.forEach(function(v, i){
                    // generate position from theta
                    let theta_i = theta;
                    if (t.spawn_outside) {
                        var x_i = Math.round(Z[1] + 1.15 * r * Math.cos(theta_i));
                        var y_i = Math.round(Z[0] + 1.15 * r * Math.sin(theta_i));
                    } else {
                        var x_i = Math.round(Z[1] + 0.8 * Math.random() * r * Math.cos(theta_i));
                        var y_i = Math.round(Z[0] + 0.8 * Math.random() * r * Math.sin(theta_i));
                    }
                    let color = '#000;';
                    
                    // check if item is in vocabulary already
                    if ((v in vocabulary) & (t.freeze_words)) {
                        // if yes, we can use its last known location
                        x_i = vocabulary[v][0];
                        y_i = vocabulary[v][1];
                        color = '#336;'
                    }

                    // place item
                    var item = $('<div id="circle-context-item-{0}" style="position: absolute; left: {1}px; top: {2}px; color: {3};" class="circle-context-item" previously-moved="false"><button style="margin: none; border: none; background: none; padding: none; color: {3}; font-size: 12px;">{4}</button></div>'.format(i, x_i, y_i, color, v));
                    item.appendTo(taskContainer);
                    
                    // if required, reposition item around centre
                    if ((t.freeze_words == false) || ((v in vocabulary) == false && (t.freeze_words == true))) {
                        x_i = x_i - item.width() / 2;
                        y_i = y_i - item.height() / 2;
                        item.css({'left': `${x_i}px`, 'top': `${y_i}px`});
                    }

                    // add animation
                    if (t.gamify) {
                        item.hide();

                        setTimeout(function(){
                            item.fadeIn(1000);
                            if (t.sound) {
                                playSound(t.sound_fx_down);
                            }
                        }, 500 + 50 * i);
                    }

                    // record initial position of item
                    let xy = pixelsToCoord(x_i + item.width() / 2, y_i + item.height() / 2, Z, r);
                    trial_initial[v] = xy;

                    // roll theta
                    theta -= delta;
                });

                // setup time tracker
                let interval = null;
                function startTimeProgressBar(durationMs = 5 * 60 * 1000) {
                    const $fill = $('#progress-time');
                    const $label = $('#progress-time-label');
                    const startTime = Date.now();
                    let sound_played = false;
                  
                    interval = setInterval(() => {
                      const elapsed = Date.now() - startTime;
                      const percent = Math.min((elapsed / durationMs) * 100, 100);
                      $fill.css('width', percent + '%');
                  
                      // visual warning for final minute
                      if (elapsed >= durationMs - 60000) {
                        $fill.addClass('time-warning');
                      }

                      // visual warning for last 30 seconds
                      if (elapsed >= durationMs - 30000) {
                        $fill.addClass('time-warning-harsh');
                      }
                  
                      // Optional: update label to mm:ss
                      const remainingMs = Math.max(durationMs - elapsed, 0);
                      const minutes = Math.floor(remainingMs / 60000);
                      const seconds = Math.floor((remainingMs % 60000) / 1000);
                      $label.text(`${lang[t.study_language].time}: ${minutes}:${seconds.toString().padStart(2, '0')}`);
                  
                      if (elapsed >= durationMs) {
                        clearInterval(interval);
                        if (t.sound) playSound(t.sound_fx_alarm); // play alarm sound, time is up
                    
                        const $text = $('<div class="animated-text"></div>').text(lang[t.study_language].time_up).css({
                            position: 'absolute',
                            top: '50%',
                            left: '50%',
                            transform: 'translate(-50%, -50%)',
                            fontSize: '16px',
                            fontWeight: 'bold',
                            background: '#4b4237',
                            color: '#c8b6a6',
                            zIndex: 9999,
                            pointerEvents: 'none',
                            opacity: 0
                          });
                        
                          $('body').append($text);
                          
                          // Animate bump in
                          $text.animate({
                            opacity: 1,
                            top: '47%'
                          }, 300, 'swing', function () {
                            // Hold position briefly
                            setTimeout(() => {
                              // Animate out: slide down & fade
                              $text.animate({
                                opacity: 0,
                                top: '53%'
                              }, 400, 'swing', function () {
                                $text.remove();
                              });
                            }, 4000);
                          });
                      }
                    }, 1000);
                  
                    return interval; // return so you can clear it if needed
                }

                // setup time tracker for submit button
                function startTimeSubmit(durationMs = 2.5 * 60 * 1000) {
                    const $button = $('#circle-task-submit');
                    const startTime = Date.now();
                    
                    const interval = setInterval(() => {
                      const elapsed = Date.now() - startTime;
                      const remainingMs = Math.max(durationMs - elapsed, 0);
                      const minutes = Math.floor(remainingMs / 60000);
                      const seconds = Math.floor((remainingMs % 60000) / 1000);
                      $button.html(`${lang[t.study_language].complete} (${lang[t.study_language].complete_in} ${minutes}:${seconds.toString().padStart(2, '0')})`);
                  
                      if (elapsed >= durationMs) {
                        clearInterval(interval);
                        $button.html(`${lang[t.study_language].complete}`);
                        $button.removeAttr('disabled').addClass('enabled');
                      }
                    }, 1000);
                  
                    return interval; // return so you can clear it if needed
                }

                // setup timers
                let timer_interval = startTimeProgressBar(t.time_per_circle);
                let submit_interval = startTimeSubmit(t.min_per_circle);

                // setup reference to active item
                let active = undefined;

                // setup drag n drop
                $('.circle-context-item').on('mousedown', function(event){
                    // boiler plate
                    event.preventDefault();

                    // grab item
                    let item = $(this);
                    active = $(this);

                    // gamify
                    if (t.gamify) {
                        item.find('button').animate({'font-size': '20px'}, 250);
                    }

                    if (t.sound) {
                        playSound(t.sound_fx_up);
                    }

                    // log event
                    const left = parseFloat(item.css('left'));
                    const top = parseFloat(item.css('top'));
                    const width = item.outerWidth();
                    const height = item.outerHeight();
                    
                    const centerX = left + width / 2;
                    const centerY = top + height / 2;

                    const word = item.find('button').html();
                    const pos = pixelsToCoord(centerX, centerY, Z, r);

                    actions.push({ type: 'drag', word: word, pos: pos, time: performance.now() - actions_start_time });
                    
                    // grab bounding shift
                    const shiftX = event.clientX - item.offset().left;
                    const shiftY = event.clientY - item.offset().top;
                    
                    // setup movement
                    let moveTo = function(x, y) {
                        // Get current size and center of the dragged item
                        const itemWidth = item.outerWidth();
                        const itemHeight = item.outerHeight();
                        
                        const itemLeft = x - shiftX;
                        const itemTop = y - shiftY;
                    
                        const itemCenterX = itemLeft + itemWidth / 2;
                        const itemCenterY = itemTop + itemHeight / 2;
                    
                        // Circle center and radius
                        const circleCenterX = Z[1];
                        const circleCenterY = Z[0];
                    
                        const dx = itemCenterX - circleCenterX;
                        const dy = itemCenterY - circleCenterY;
                        const distanceFromCenter = Math.sqrt(dx * dx + dy * dy);
                        const angle = Math.atan2(dy, dx);
                    
                        // Previously moved flag
                        const attr = item.attr('previously-moved');
                    
                        let nx, ny;
                    
                        if ((distanceFromCenter > r2) && ((typeof attr !== 'undefined' && attr !== "false") || !t.spawn_outside)) {
                            // Clamp the element so its center is at the circle border
                            const clampedCenterX = circleCenterX + Math.cos(angle) * (r2 - itemWidth / 2);
                            const clampedCenterY = circleCenterY + Math.sin(angle) * (r2 - itemHeight / 2);
                    
                            // Adjust so top-left corner is positioned correctly
                            nx = Math.round(clampedCenterX - itemWidth / 2);
                            ny = Math.round(clampedCenterY - itemHeight / 2);
                        } else {
                            // Move freely
                            nx = itemLeft;
                            ny = itemTop;
                        }
                    
                        // Apply the new position
                        item.css({
                            left: `${nx}px`,
                            top: `${ny}px`
                        });
                    
                        // Optional: gamification scaling effect
                        if (t.gamify) {
                            items.forEach(function(v, i){
                                const other = $(`#circle-context-item-${i}`);
                                if (active.attr('id') === other.attr('id')) return;
                                if (other.attr('previously-moved') === undefined || other.attr('previously-moved') === 'false') return;
                    
                                const x_o = other.offset().left + other.outerWidth() / 2;
                                const y_o = other.offset().top + other.outerHeight() / 2;
                    
                                const dist = ((x_o - itemCenterX) ** 2 + (y_o - itemCenterY) ** 2) ** 0.5 / d;
                                let size = (1 - Math.pow(dist, 1.25)) * 16;
                                size = Math.min(Math.max(size, 10), 16);
                    
                                other.find('button').stop(true, true).animate({'font-size': `${size}px`}, 5);
                            });
                        }
                    };
                    
                    // move
                    moveTo(event.pageX, event.pageY);

                    // setup movement tracking
                    $(document).on('mousemove.dragDrop', function(event){
                        moveTo(event.pageX, event.pageY);
                    });
                });

                $(document).on('mouseup', function(event) {
                    $(document).off('mousemove.dragDrop');
                
                    if (typeof(active) !== 'undefined') {
                        // Gamify: Reset sizes
                        if (t.gamify) {
                            var progress = 1;

                            items.forEach(function(v, i) {
                                let other = $(`#circle-context-item-${i}`);
                                if (other.attr('previously-moved') === 'true') progress += 1;
                                if (active.attr('id') === other.attr('id')) return;
                
                                other.find('button').animate({'font-size': '12px'}, 250);
                            });
                
                            active.find('button').animate({'font-size': '12px'}, 250);
                            
                            progress = progress / items.length * 100;
                            $('#progress-this').css('width', `${progress}%`);
                        }

                        if (t.sound) {
                            playSound(t.sound_fx_down);
                        }
                
                        // recheck boundary if spawn_outside is allowed
                        if (t.spawn_outside) {
                            const x = parseFloat(active.css('left'));
                            const y = parseFloat(active.css('top'));
                
                            const itemWidth = active.outerWidth();
                            const itemHeight = active.outerHeight();
                
                            const itemCenterX = x + itemWidth / 2;
                            const itemCenterY = y + itemHeight / 2;
                
                            const dx = itemCenterX - Z[1]; // delta X from circle center
                            const dy = itemCenterY - Z[0]; // delta Y from circle center
                            const distanceFromCenter = Math.sqrt(dx * dx + dy * dy);
                
                            if (distanceFromCenter > r2) {
                                const angle = Math.atan2(dy, dx);
                
                                // Clamp center to circle border (minus half size)
                                const clampedCenterX = Z[1] + Math.cos(angle) * (r2 - itemWidth / 2);
                                const clampedCenterY = Z[0] + Math.sin(angle) * (r2 - itemHeight / 2);
                
                                const nx = Math.round(clampedCenterX - itemWidth / 2);
                                const ny = Math.round(clampedCenterY - itemHeight / 2);
                
                                active.css({'left': `${nx}px`, 'top': `${ny}px`});
                            }
                        }

                        // log event
                        const left = parseFloat(active.css('left'));
                        const top = parseFloat(active.css('top'));
                        const width = active.outerWidth();
                        const height = active.outerHeight();
                        
                        const centerX = left + width / 2;
                        const centerY = top + height / 2;

                        const word = active.find('button').html();
                        const pos = pixelsToCoord(centerX, centerY, Z, r);

                        actions.push({ type: 'drop', word: word, pos: pos, time: performance.now() - actions_start_time });
                        
                        if (t.gamify) {
                            // particle animation
                            const x_i = active.offset().left + active.outerWidth() / 2;
                            const y_i = active.offset().top + active.outerHeight() / 2;
                            const t_x = $('#progress-this').offset().left + $('#progress-this').outerWidth() / 2;
                            const t_y = $('#progress-this').offset().top + $('#progress-this').outerHeight() / 2;
                        
                            const numParticles = 6;
                            const duration = 600;
                        
                            for (let i = 0; i < numParticles; i++) {
                                // Random spawn offset
                                const spawnOffsetX = (Math.random() - 0.5) * 30;
                                const spawnOffsetY = (Math.random() - 0.5) * 30;
                        
                                // Random size
                                const size = 4 + Math.random() * 4;
                        
                                // Create particle
                                const $particle = $('<div class="particle"></div>').css({
                                    left: (x_i + spawnOffsetX) + 'px',
                                    top: (y_i + spawnOffsetY) + 'px',
                                    width: size + 'px',
                                    height: size + 'px'
                                });
                        
                                $particle.appendTo(taskContainer);
                        
                                // Start animation
                                animateParticleRandomFlight(
                                    $particle,
                                    x_i + spawnOffsetX,
                                    y_i + spawnOffsetY,
                                    t_x,
                                    t_y,
                                    duration
                                );
                            }
                        }

                        // Final state update
                        active.find('button').css({'color': '#666'});
                        active.attr('previously-moved', 'true');
                        active = undefined;
                    }
                });

                // setup submission handler
                let pSubmit = function(){
                    // disable
                    submit.off('click');

                    // ask to confirm
                    if (t.skip_confirmation || confirm(lang[t.study_language].confirmation)) {
                        var successful = true;
                        var final = {};
                        
                        // prepare submission
                        $('.circle-context-item').each(function(i) {
                            const item = $(this);
                        
                            // Compute item's center coordinates
                            const left = parseFloat(item.css('left'));
                            const top = parseFloat(item.css('top'));
                            const width = item.outerWidth();
                            const height = item.outerHeight();
                        
                            const centerX = left + width / 2;
                            const centerY = top + height / 2;
                        
                            const dx = centerX - Z[1];
                            const dy = centerY - Z[0];
                            const distanceFromCenter = Math.sqrt(dx * dx + dy * dy);
                        
                            // Check if the center is within the circle (with margin if needed)
                            const margin = 1.0;
                            if (distanceFromCenter > r2 * margin) {
                                successful = false;
                                if (t.gamify) {
                                    shakeWord(item);
                                } else {
                                    item.find('button').css({color: '#600'});
                                }
                            } else {
                                // Valid placement
                                const word = item.find('button').html();
                                const pos = pixelsToCoord(centerX, centerY, Z, r);
                        
                                final[word] = pos;
                                vocabulary[word] = [left.toFixed(2), top.toFixed(2)];
                            }
                        });

                        // reset if unsuccessful
                        if (!successful) {
                            final = {};
                            submit.on('click', pSubmit);
                            if (t.sound) playSound(t.sound_fx_error);
                            return;
                        } else {
                            if (t.sound) playSound(t.sound_fx_success);
                        }
                        
                        // remove timer interval
                        clearInterval(timer_interval);
                        clearInterval(submit_interval);
                        clearInterval(interval);
                        clearInterval(mouse_interval);
                        
                        // copy data to final array
                        trial_final = final;

                        // add to data array
                        data.boards.initial.push(trial_initial);
                        data.boards.actions.push(actions);
                        data.boards.final.push(trial_final);
                        data.boards.rt.push(performance.now() - actions_start_time);

                        console.log(actions);
                        console.log(data.boards.rt);
                        
                        let pFinishThis = function(){
                            // finalise trial or task
                            if ((current_trial + 1) >= circles.length) {
                                // we just completed the last trial!
                                // stringify data
                                data.boards = JSON.stringify(data.boards);
                                
                                // add rt
                                data.rt = Date.now() - time_start;

                                // finalise trial
                                jsPsych.pluginAPI.clearAllTimeouts();
                                jsPsych.finishTrial(data);
                            } else {
                                // we haver more trials to complete
                                // check breaks settings
                                if (t.circles_per_break > 0) {
                                    // breaks desired, so check for break
                                    if ((current_trial + 1) % t.circles_per_break == 0) {
                                        // insert a break here before we continue
                                        const $text = $('<div class="animated-text"></div>').text(lang[t.study_language].break.format(current_trial + 1, circles.length)).css({
                                            position: 'absolute',
                                            top: '50%',
                                            left: '50%',
                                            transform: 'translate(-50%, -50%)',
                                            fontSize: '16px',
                                            fontWeight: 'bold',
                                            background: '#4b4237',
                                            color: '#c8b6a6',
                                            zIndex: 9999,
                                            pointerEvents: 'none',
                                            opacity: 0
                                        });
                                        
                                        $('body').append($text);

                                        // animate
                                        $text.animate({
                                            opacity: 1,
                                            top: '47%'
                                        }, 300, 'swing', function(){});
                                        
                                        // setup submit button
                                        submit.on('click', function(){
                                            // animate
                                            $text.animate({
                                                opacity: 0,
                                                top: '53%'
                                            }, 400, 'swing', function(){
                                                $text.remove();
                                                run_trial(current_trial + 1);
                                            });
                                        });
                                    } else {
                                        // no break required, commence next trial
                                        run_trial(current_trial + 1);
                                    }
                                } else {
                                    // no breaks required
                                    run_trial(current_trial + 1);
                                }                                
                            }
                        };

                        // Gamify: circle implosion
                        if (t.gamify) {
                            function animateCircleTrace(canvas, duration = 2000) {
                                const ctx = canvas.getContext('2d');
                                const r = canvas.width / 2 - 1;
                                const centerX = r + 1;
                                const centerY = r + 1;
                                const startTime = performance.now();
                              
                                function drawFrame(time) {
                                  const elapsed = time - startTime;
                                  const progress = Math.min(elapsed / duration, 1);
                                  const endAngle = progress * 2 * Math.PI;
                              
                                  // Clear only the border overlay (optional if using separate layer)
                                  ctx.clearRect(0, 0, canvas.width, canvas.height);
                                  
                                  // Base circle fill (optional if already drawn)
                                  ctx.beginPath();
                                  ctx.arc(centerX, centerY, r, 0, 2 * Math.PI);
                                  ctx.fillStyle = `rgb(200, 200, 200)`;
                                  ctx.fill();
                              
                                  // Tracing arc
                                  ctx.beginPath();
                                  ctx.arc(centerX, centerY, r, 0, endAngle);
                                  ctx.strokeStyle = '#a3b18a';
                                  ctx.lineWidth = 5;
                                  ctx.stroke();
                              
                                  if (progress < 1) {
                                    requestAnimationFrame(drawFrame);
                                  }
                                }
                              
                                requestAnimationFrame(drawFrame);
                            }

                            function collapseCircle(duration = 1000, fps = 60) {
                                const canvas = $('#circle-context').get(0);
                                const context = canvas.getContext("2d");
                                const r_initial = canvas.width / 2 - 1;
                                const centerX = r_initial + 1;
                                const centerY = r_initial + 1;
                            
                                let frame = 0;
                                const totalFrames = duration / (1000 / fps);
                            
                                function animate() {
                                    const progress = frame / totalFrames;
                                    const r = r_initial * (1 - progress);
                            
                                    // Clear the canvas
                                    context.clearRect(0, 0, canvas.width, canvas.height);
                            
                                    // Draw smaller circle
                                    context.beginPath();
                                    context.arc(centerX, centerY, r, 0, 2 * Math.PI);
                                    context.fillStyle = `rgb(200, 200, 200)`;
                                    context.fill();
                                    context.stroke();
                            
                                    frame++;
                                    if (frame <= totalFrames) {
                                        requestAnimationFrame(animate);
                                    }
                                }
                            
                                animate();
                            }

                            animateCircleTrace($('#circle-context')[0], duration = 1750);

                            items.forEach(function(v, i){
                                const item = $(`#circle-context-item-${i}`);
                                setTimeout(function(){
                                    item.fadeOut(500)
                                }, 50 * i);
                            });

                            setTimeout(function(){
                                if (t.sound) playSound(t.sound_fx_wheesh);
                                collapseCircle(duration = 500, fps = 60)
                                
                                setTimeout(function(){
                                    const x_i = $('#progress-this').offset().left + $('#progress-this').outerWidth() / 2;
                                    const y_i = $('#progress-this').offset().top + $('#progress-this').outerHeight() / 2;
                                    const t_x = $('#progress-all').offset().left + $('#progress-all').outerWidth() / 2;
                                    const t_y = $('#progress-all').offset().top + $('#progress-all').outerHeight() / 2;
                                
                                    const numParticles = 64;
                                    const dur = 600;
                                
                                    for (let i = 0; i < numParticles; i++) {
                                        // Random spawn offset
                                        const spawnOffsetX = (Math.random() - 0.5) * 30;
                                        const spawnOffsetY = (Math.random() - 0.5) * 30;
                                
                                        // Random size
                                        const size = 4 + Math.random() * 4;
                                
                                        // Create particle
                                        const $particle = $('<div class="particle"></div>').css({
                                            left: (x_i + spawnOffsetX) + 'px',
                                            top: (y_i + spawnOffsetY) + 'px',
                                            width: size + 'px',
                                            height: size + 'px'
                                        });
                                
                                        $particle.appendTo(taskContainer);
                                
                                        // Start animation
                                        animateParticleRandomFlight(
                                            $particle,
                                            x_i + spawnOffsetX,
                                            y_i + spawnOffsetY,
                                            t_x,
                                            t_y,
                                            dur
                                        );
                                    }
                                }, 500);
                            }, 2000);
                            setTimeout(pFinishThis, 3100);
                        } else {
                            pFinishThis();
                        }
                    } else {
                        // resume
                        submit.on('click', pSubmit);
                    }
                };

                submit.on('click', pSubmit);
            };

            // start trial
            let time_start = Date.now();
            pTrial();
        }

        // set loading
        e.innerHTML = '';
        let loading = $('<p>' + lang[t.study_language].loading + '...</p>');
        loading.appendTo(e);

        // create circles
        var circles;
        if (t.circles.length == 0) {
            circles = getCoveringCircles(t.words, t.per_circle, t.max_attempts);
        } else {
            circles = t.circles;
        }

        // add anchors, if required
        for (let i = 0; i < circles.length; i++) {
            for (let j = 0; j < t.anchors.length; j++) {
                if (circles[i].includes(t.anchors[j]) == false) {
                    circles[i].push(t.anchors[j]);
                }
            }
        }

        // run first batch
        run_trial(0);
    }

    return plugin;
})();