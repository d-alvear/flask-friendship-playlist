// client-side js
// run by the browser each time your view template is loaded

$(function() {

  $('form').submit(function(event) {
    event.preventDefault();
    
    $('#new-releases').empty();
    let country = $('select').val();
    
    // Send a request to our backend (server.py) to get new releases for the currently selected country
    $.get('/new_releases?' + $.param({country: country}), function(new_releases) {
      
      // Loop through each album in the list
      new_releases.albums.items.forEach(function(release) {
        
        // Use the returned information in the HTML
        let div = $('<div class="sp-entity-container"><a href="' + release.external_urls.spotify + 
                '"><div style="background:url(\'' + release.images[0].url + 
                '\')" class="sp-cover" alt="Album cover"></div></a><h3 class="sp-title">' + release.name + 
                '</h3><p class="text-grey-55 sp-by">By ' + release.artists[0].name + '</p></div>')
        
        div.appendTo('#new-releases')
        
      });
    });
  });

});
