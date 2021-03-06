#
#   Copyright 2007-2015 The OpenMx Project
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
# 
#        http://www.apache.org/licenses/LICENSE-2.0
# 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


unlockCounter <- function() {
	if(.untitledLocked) {
		namespace <- getNamespace("OpenMx")
		unlockBinding('.untitledLocked', namespace)
		.untitledLocked <<- FALSE
		lockBinding('.untitledLocked', namespace)
		unlockBinding('.untitledCounter', namespace)
	}
}

imxUntitledNumberReset <- function() {
	unlockCounter()
	.untitledCounter <<- 0
}

imxUntitledNumber <- function() {
	unlockCounter()
	.untitledCounter <<- .untitledCounter + 1
	return(.untitledCounter)	
}

imxUntitledName <- function() {
	name <- paste("untitled", format(imxUntitledNumber(), scientific = FALSE),
		sep="")
	return(name)
}

.untitledCounter <- 0
.untitledLocked <- TRUE
