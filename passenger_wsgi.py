"""
Phusion Passenger entry point (shim).

The Flask app lives in app.py. Plain Passenger imports this file and looks
for `application`; cPanel's "Setup Python App" ignores it and writes its own
stub here that loads the configured startup file - which must be app.py, not
passenger_wsgi.py, or the stub loads itself until RecursionError.
See DEPLOYMENT.md, section 4.
"""

from app import application  # noqa: F401
