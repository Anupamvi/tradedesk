PROJECT_NAME=HealthAgeApp
PROJECT_FILE=$(PROJECT_NAME).xcodeproj
SCHEME=$(PROJECT_NAME)

.PHONY: generate open build simulator

generate:
	xcodegen generate

open: generate
	open $(PROJECT_FILE)

build: generate
	xcodebuild -project $(PROJECT_FILE) -scheme $(SCHEME) -destination 'platform=iOS Simulator,name=iPhone 16' build

simulator: generate
	xcodebuild -project $(PROJECT_FILE) -scheme $(SCHEME) -destination 'platform=iOS Simulator,name=iPhone 16' build
