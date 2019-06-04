// store user selected model
var modelType
// function to retrieve user selected model
function inputChangeModelType(e) {
  modelType = document.getElementById("selected_model").value
}
// store user selected month
var modelMonth
// function to retrieve user selected month
function inputChangeMonth(e) {
  modelMonth = document.getElementById("selected_month").value
}
// store user selected day
var modelDay
// function to retrieve user selected day
function inputChangeDay(e) {
  modelDay = document.getElementById("selected_day").value
}
// store user selected year
var modelYear
// function to retrieve user selected year
function inputChangeYear(e) {
  modelYear = document.getElementById("selected_year").value
}


// store end variables
// Previous day's DFF
var Y_prev
// Predicted probability that the DFF of the day of interest going up
var Y_prob_a
// Predicted probability that the DFF of the day of interest keep the same
var Y_prob_b
// Predicted probability that the DFF of the day of interest going down
var Y_prob_c
// DFF of the day of interest
var Y_to_pred
// is the date
var Date1

function returnData(e){

  fetch("/predict/"+modelType+"/"+modelMonth+"/"+modelDay+"/"+modelYear, function(z){})
    .then(response => response.json())
    .then(data => {
    Y_prev = data.Y_prev;
    Y_prob_a = data.Y_prob[0].toFixed(3);
    Y_prob_b = data.Y_prob[1].toFixed(3);
    Y_prob_c = data.Y_prob[2].toFixed(3);
    Y_to_pred = data.Y_to_pred;
    Date1 = data.date;
    console.log("Selected date: " + Date1);
    console.log("The DFF of the selected date: " + Y_to_pred);
    console.log("The previous day's DFF Value: " + Y_prev);
    console.log("Our predicted probability the DFF will go up: " + Y_prob_a);
    console.log("Our predicted probability the DFF will stay the same: " + Y_prob_b);
    console.log("Our predicted probability the DFF will go down: " + Y_prob_c);

    // chart 1
    var Date2 = new Date(Date1);
    Date2.setDate(Date2.getDate() - 1);
    Date2 = (Date2.getMonth() + 1) + '/' + Date2.getDate() + '/' +  Date2.getFullYear();
    console.log("Selected date: " + Date2);

    var
        Y_Axis_DFF_Data = [Y_prev, Y_to_pred],
        X_Axis_Date_Data = [Date2, Date1];

    // Chart 2
    var
        Y_Axis_DFF_Data2 = [Y_prob_a, Y_prob_b, Y_prob_c],
        X_Axis_Date_Data2 = ["go up", "stay the same", "go down"];



    function makeResponsive() {

      // if the SVG area isn't empty when the send button is clicked,
      // remove it
      var svgArea = d3.select("#chart").select("svg");

      if (!svgArea.empty()) {
          svgArea.remove();
      }

      // svg params
      var svgHeight = 275;
      var svgWidth = 350;

      // margins
      var margin = {
          top: 50,
          right: 50,
          bottom: 50,
          left: 50
      };

      // chart area minus margins
      var chartHeight = svgHeight - margin.top - margin.bottom;
      var chartWidth = svgWidth - margin.left - margin.right;

      // create svg container
      var svg = d3.select("#chart").append("svg")
          .attr("height", svgHeight)
          .attr("width", svgWidth);

      // shift everything over by the margins
      var chartGroup = svg.append("g")
          .attr("transform", `translate(${margin.left}, ${margin.top})`);

      var y_domain_max = (d3.max(Y_Axis_DFF_Data)*1.50).toFixed(3);
      var y_domain_min = (d3.max(Y_Axis_DFF_Data)/1.50).toFixed(3);

      // scale y to chart height
      var yScale = d3.scaleLinear()
          .domain([y_domain_min, y_domain_max])
          .range([chartHeight, 0]);

      // scale x to chart width
      var xScale = d3.scaleBand()
          .domain(X_Axis_Date_Data)
          .range([0, chartWidth])
          .padding(0.1);

      // create axes
      var yAxis = d3.axisLeft(yScale);
      var xAxis = d3.axisBottom(xScale);

      // set x to the bottom of the chart
      chartGroup.append("g")
          .attr("transform", `translate(0, ${chartHeight})`)
          .call(xAxis);

      // set y to the y axis
      chartGroup.append("g")
          .call(yAxis);

      chartGroup.selectAll("circle")
          .data(Y_Axis_DFF_Data)
          .enter()
          .append("circle")
          .attr("cx", (d, i) => xScale(X_Axis_Date_Data[i]) + 55)
          .attr("cy", d => yScale(d))
          .attr("r", 6)
          // .attr("height", d => chartHeight - yScale(d))
          .attr("fill", "teal")
          // event listener for onclick event
          .on("click", function(d, i) {
            alert(`DFF: ${Y_Axis_DFF_Data}`);
          });

        // Add the text label for the Y axis
    chartGroup.append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", 0 - margin.left)
        .attr("x",0 - (chartHeight / 2))
        .attr("dy", "1em")
        .style("text-anchor", "middle")
        .text("Fed Rate (%)");

        // Add the text label for the x axis
    chartGroup.append("text")
        .attr("transform", "translate(" + (chartWidth / 2) + " ," + ((chartHeight - 5) + margin.bottom) + ")")
        .style("text-anchor", "middle")
        .text("Date");

      // Chart 2
      // if the SVG2 area isn't empty...
      var svgArea2 = d3.select("#chart2").select("svg");

      if (!svgArea2.empty()) {
          svgArea2.remove();
      }

      // svg2 params
      var svgHeight2 = 275;
      var svgWidth2 = 350;

      // margins
      var margin2 = {
          top: 50,
          right: 50,
          bottom: 50,
          left: 50
      };

      // chart area minus margins
      var chartHeight2 = svgHeight2 - margin2.top - margin2.bottom;
      var chartWidth2 = svgWidth2 - margin2.left - margin2.right;

      // create svg2 container
      var svg2 = d3.select("#chart2").append("svg")
          .attr("height", svgHeight2)
          .attr("width", svgWidth2);

      // shift everything over by the margins
      var chartGroup2 = svg2.append("g")
          .attr("transform", `translate(${margin2.left}, ${margin2.top})`);

      // scale y to chart height
      var yScale2 = d3.scaleLinear()
          .domain([0, d3.max(Y_Axis_DFF_Data2)])
          .range([chartHeight2, 0]);

      // scale x to chart width
      var xScale2 = d3.scaleBand()
          .domain(X_Axis_Date_Data2)
          .range([0, chartWidth2])
          .padding(0.1);

      // create axes2
      var yAxis2 = d3.axisLeft(yScale2);
      var xAxis2 = d3.axisBottom(xScale2);

      // set x to the bottom of the chart
      chartGroup2.append("g")
          .attr("transform", `translate(0, ${chartHeight2})`)
          .call(xAxis2);
          // .text("probability")

      // set y to the y axis
      chartGroup2.append("g")
          .call(yAxis2);

      chartGroup2.selectAll("rect")
          .data(Y_Axis_DFF_Data2)
          .enter()
          .append("rect")
          .attr("x", (d, i) => xScale2(X_Axis_Date_Data2[i]))
          .attr("y", chartHeight2)
          .transition().duration(5000)
          .ease(d3.easeExp)
          .attr("y", d => yScale2(d))
          .attr("width", xScale2.bandwidth())
          .attr("height", d => chartHeight2 - yScale2(d))
          // .attr("height", d => chartHeight - yScale2(d))
          .attr("fill", "teal")
          // event listener for onclick event
          .on("click", function(d, i) {
            alert(`DFF: ${Y_Axis_DFF_Data2}`);
          })

          // Add the text label for the Y axis
      chartGroup2.append("text")
          .attr("transform", "rotate(-90)")
          .attr("y", 0 - margin2.left)
          .attr("x",0 - (chartHeight2 / 2))
          .attr("dy", "1em")
          .style("text-anchor", "middle")
          .text("Probability (%)");

          // Add the text label for the x axis
      chartGroup2.append("text")
          .attr("transform", "translate(" + (chartWidth2 / 2) + " ," + ((chartHeight2 - 5) + margin2.bottom) + ")")
          .style("text-anchor", "middle")
          .text("Model Prediction");

    }
    makeResponsive();
      })
}

(async function () {
  const data = [{
      date: '2003',
      value: 1.12,
    },
    {
      date: '2004',
      value: 1.35,
    },
    {
      date: '2005',
      value: 3.22,
    },
    {
      date: '2006',
      value: 4.96,
    },
    {
      date: '2007',
      value: 5.02,
    },
    {
      date: '2008',
      value: 1.92,
    },
    {
      date: '2009',
      value: 0.15,
    },
    {
      date: '2010',
      value: 0.17,
    },
    {
      date: '2011',
      value: 0.10,
    },
    {
      date: '2012',
      value: 0.14,
    },
    {
      date: '2013',
      value: 0.10,
    },
    {
      date: '2014',
      value: 0.09,
    },
    {
      date: '2015',
      value: 0.13,
    },
    {
      date: '2016',
      value: 0.39,
    },
    {
      date: '2017',
      value: 1.00,
    },
    {
      date: '2018',
      value: 1.83,
    },
    {
      date: '2019',
      value: 2.40,
    }
  ]
  // Define SVG area dimensions
  const
    svgWidth = 375,
    svgHeight = 300;

  // Define the chart's margins as an object
  const margin = {
    top: 60,
    right: 60,
    bottom: 60,
    left: 60
  };

  // Define dimensions of the chart area
  const chartWidth = svgWidth - margin.left - margin.right;
  const chartHeight = svgHeight - margin.top - margin.bottom;

  // Select body, append SVG area to it, and set its dimensions
  const svg = d3.select("#line")
    .append("svg")
    .attr("width", svgWidth)
    .attr("height", svgHeight);

  // Append a group area, then set its margins
  const chartGroup = svg.append("g")
    .attr("transform", `translate(${margin.left}, ${margin.top})`);

  // Configure a parseTime function which will return a new Date object from a string
  const parseTime = d3.timeParse("%Y");

  // Format the date and cast the force value to a number
  data.forEach(function (data) {
    data.date = parseTime(data.date);
    data.value = +data.value;
  });

  // Configure a time scale
  // d3.extent returns the an array containing the min and max values for the property specified
  const xTimeScale = d3.scaleTime()
    .domain(d3.extent(data, d => d.date))
    .range([0, chartWidth]);

  // Configure a linear scale with a range between the chartHeight and 0
  const yLinearScale = d3.scaleLinear()
    .domain([0, d3.max(data, d => d.value)])
    .range([chartHeight, 0]);

  // Create two new functions passing the scales in as arguments
  // These will be used to create the chart's axes
  const bottomAxis = d3.axisBottom(xTimeScale);
  const leftAxis = d3.axisLeft(yLinearScale);

  // Configure a line function which will plot the x and y coordinates using our scales
  var drawLine = d3.line()
    .x(data => xTimeScale(data.date))
    .y(data => yLinearScale(data.value))
  // create the line path
  var path = chartGroup.append('path')
    .attr('d', drawLine(data))
    .attr('stroke', 'teal')
    .attr('stroke-width', 4)
    .attr('fill', 'none')
  // Add a curtain to hide the chart until transition
  var curtain = chartGroup.append('rect')
    .attr('x', -1 * chartWidth)
    .attr('y', -1 * chartHeight)
    .attr('height', 1.25 * chartHeight)
    .attr('width', chartWidth)
    .attr('class', 'curtain')
    .attr('transform', 'rotate(180)')
    .style('fill', '#ffffff')
  // Apply animation to line
  curtain.transition()
    .duration(15000)
    .attr('x', -2 * chartWidth)

  // Append an SVG group element to the chartGroup, create the left axis inside of it
  chartGroup.append("g")
    .classed("axis", true)
    .call(leftAxis);

  // Append an SVG group element to the chartGroup, create the bottom axis inside of it
  // Translate the bottom axis to the bottom of the page
  chartGroup.append("g")
    .classed("axis", true)
    .attr("transform", `translate(0, ${chartHeight})`)
    .call(bottomAxis);

  chartGroup.append("text")
    .attr("transform", "rotate(-90)")
    .attr("y", 0 - margin.left)
    .attr("x",0 - (chartHeight / 2))
    .attr("dy", "1em")
    .style("text-anchor", "middle")
    .text("Fed Rate (%)");

    // Add the text label for the x axis
  chartGroup.append("text")
    .attr("transform", "translate(" + (chartWidth / 2) + " ," + ((chartHeight - 5) + margin.bottom) + ")")
    .style("text-anchor", "middle")
    .text("Year");

})()