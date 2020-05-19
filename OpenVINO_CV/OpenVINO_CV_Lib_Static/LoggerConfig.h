#pragma once

#include <log4cplus/logger.h>
#include <log4cplus/initializer.h>
#include <log4cplus/configurator.h>
#include <log4cplus/loggingmacros.h>

#include <log4cplus/appender.h>
#include <log4cplus/fileappender.h>
#include <log4cplus/socketappender.h>
#include <log4cplus/syslogappender.h>
#include <log4cplus/consoleappender.h>
#include <log4cplus/log4judpappender.h>

using namespace log4cplus;


/* global ��־��¼���� */
log4cplus::Logger logger;

/* log4cplus ���ã���ȡ log4cplus.config �����ļ� */
static void Log4CplusConfigure()
{
	static log4cplus::Initializer initializer;
	static log4cplus::PropertyConfigurator property(LOG4CPLUS_TEXT2("log4cplus.config"));
	property.configure();

	logger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT2("global"));
	LOG4CPLUS_INFO(logger, "log4cplus conifgure complete.");

	//log4cplus::Logger::shutdown();
}
	