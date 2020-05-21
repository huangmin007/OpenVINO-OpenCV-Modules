#pragma once

#include <iostream>
#include <string>

namespace slog 
{
    class LogStreamEndLine { };
    static constexpr LogStreamEndLine endl;

    class LogStreamBoolAlpha { };
    static constexpr LogStreamBoolAlpha boolalpha;


    class LogStream 
    {
        std::string _prefix;
        std::ostream* _log_stream;
        bool _new_line;

    public:        
        LogStream(const std::string& prefix, std::ostream& log_stream): _prefix(prefix), _new_line(true)
        {
            _log_stream = &log_stream;
        }

        
        template<class T>
        LogStream& operator<<(const T& arg) 
        {
            if (_new_line)
            {
                (*_log_stream) << "[" << std::setw(5) << std::right << _prefix << "] ";
                _new_line = false;
            }

            (*_log_stream) << arg;
            return *this;
        }

        LogStream& operator<< (const LogStreamEndLine&/*arg*/) 
        {
            _new_line = true;

            (*_log_stream) << std::endl;
            return *this;
        }

        LogStream& operator<< (const LogStreamBoolAlpha&/*arg*/) 
        {
            (*_log_stream) << std::boolalpha;
            return *this;
        }
    };


    static LogStream info("INFO", std::cout);
    static LogStream warn("WARN", std::cout);
    static LogStream error("ERROR", std::cerr);

}