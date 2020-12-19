

function validation() {
     url=document.getElementById('url_id').value;
        var regex = (/(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)/g);
        if(!regex.test(url)){
            alert("invalid URL");
            return false;
        }
       
        return true;
  }

  