// // client-side js
// // run by the browser each time your view template is loaded

// $(function() {  
//     $('form').submit(function(event) {
//       event.preventDefault();
      
//       let query = $('input').val();
//       let context = $('input[name="context"]:checked').val();
      
//       $.get('/search?' + $.param({context: context, query: query}), function(data) {
//         $('input[type="text"]').val('');
//         $('input').focus();
        
//         document.getElementById('results').innerHTML = data.tracks.items.map(track => {
//           return `<li><a href="${track.external_urls.spotify}">${track.name}   |   ${track.artists[0].name}</a></li>`;
//         }).join('\n');
//       });
//     });
  
//   });
  