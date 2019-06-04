(function($){
  $(function(){

    $('.sidenav').sidenav();
    $('.parallax').parallax();

    document.addEventListener('DOMContentLoaded', function() {
  var elems = document.querySelectorAll('.dropdown-trigger');
  var instances = M.Dropdown.init(elems, options);
});

  }); // end of document ready
})(jQuery); // end of jQuery name space
