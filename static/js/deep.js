training_data_indicies = new Array(18).fill(0);
initial_image_indicies = new Array(4).fill(0);
var number_of_times_clicked = 0;

console.log(typeof(new Date().getTime()));

$(function() {                       //run when the DOM is ready
  $(".thumbnail").click(function() {  //use a class, since your ID gets mangled

    $(this).toggleClass("thumbnail-clicked");      //add the class to the clicked element
  });
});

$('a.thumbnail').removeAttr('href');
$('#train-and-draw').removeAttr('href');

$(function() {
	$("#train-and-draw").click(function() {
        $(document.body).css({"cursor": "wait"});
        number_of_times_clicked = number_of_times_clicked + 1;

		for (var i = 0; i < training_data_indicies.length; i++) {
	    	if ($("#tdata"+String(i)).hasClass("thumbnail-clicked")) {
	    		training_data_indicies[i] = 1;
	    	} else {
	    		training_data_indicies[i] = 0;
	    	}			
		}

		for (var i = 0; i < initial_image_indicies.length; i++) {
			if ($("#iimage"+String(i)).hasClass("thumbnail-clicked")) {
				initial_image_indicies[i] = 1;
			} else {
				initial_image_indicies[i] = 0;
			}
		}

    	console.log(training_data_indicies);
    	console.log(initial_image_indicies);

        step_size = Number(d3.select("#step-size > label.active").text());

    	data_to_python = {"training_data_indicies": training_data_indicies, 
    					  "initial_image_indicies": initial_image_indicies,
                          "step_size": step_size,
                          "number_of_times_clicked": number_of_times_clicked};

    	d3.json('/run/').post(JSON.stringify(data_to_python), function(error, data) {
    			$(document.body).css({"cursor": "default"});

                console.log(data);
    			console.log(error);

                if (number_of_times_clicked == 1) {
                    d3.select("#results").append("div").attr("class", "col-md-12")
                                         .append("h2").text("Results");
                }

                var preResultsInitialImageIndex = initial_image_indicies.indexOf(1);
                var preResultsInitialImage = ['zeros', 'ones', 'noise', 'noise_blur'][preResultsInitialImageIndex];

                step_size = d3.select("#step-size > label.active").text();
                
                d3.select("#results").append("div").attr("class", "vspace-result");

                preResults = d3.select("#results").append("div").attr("class", "row");
                preResults.append("div").attr("class", "col-md-1 display-inline-block").style("width", "5%")
                          .append("h3").text("[" + String(number_of_times_clicked) + "]:");
                preResults.append("div").attr("class", "col-md-1 display-inline-block")
                          .append("a").attr("class", "thumbnail thumbnail-small thumbnail-result")
                          .append("img").attr("src", "static/images/" + preResultsInitialImage + ".png");
                preResultsHyperparameters = preResults.append("div").attr("class", "col-md-3 display-inline-block");
                preResultsHyperparameters.append("p").attr("class", "hyperparameter").text("Step-size")
                                         .append("p").attr("class", "hyperparameter").text(String(step_size));

                originalThumbBB = d3.select("#tdata0").node().getBoundingClientRect();

    			results = d3.select("#results").append("div").attr("class", "row");

    			thumbnailEnter = results.selectAll(".thumbnail-result-div").data(data.errors).enter()
		    		   .append("div").attr("class", "thumbnail-result-div")
                       .style("width", String(originalThumbBB.width + 30) + "px")
                       .style("display", "inline-block").style("padding-left", "15px").style("padding-right", "15px")
	    		       .append("a").attr("class", "thumbnail thumbnail-result");

	    		thumbnailEnter.append("img").attr("src", function(d, i) {return "static/results/" + String(number_of_times_clicked) + '_' + (i+1) + ".png" + "?v=" + String(new Date().getTime()) });
	    		thumbnailEnter.append("div").attr("class", "caption error-metric").text(function(d) { return d3.format(".2f")(d) });


    			console.log("made result row");

                $('html, body').animate({ 
                    scrollTop: $(document).height()-$(window).height()
                }, 400);

    		}
		);
	})
})

